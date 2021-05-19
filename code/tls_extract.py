import argparse
from pcap import Pcap
import scapy
from scapy.all import *
from pathlib import Path
import json
import cryptography
import pathlib
import logging
import traceback
from tqdm import tqdm
from multiprocessing import Pool
import functools
import warnings
import struct

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
conf.verb = 0

black_list = [
    "192.168.3.225",
    "192.168.7.123",
    "192.168.46.82",
    "192.168.71.160",
    "192.168.81.26",
    "192.168.116.111",
    "192.168.126.97",
    "192.168.143.111",
    "192.168.158.64",
    "192.168.192.234",
]

white_list = [
    "192.168.0.26",
    "192.168.17.214",
    "192.168.20.19",
    "192.168.38.46",
    "192.168.208.22",
    "192.168.208.178",
    "192.168.216.8",
    "192.168.226.115",
    "192.168.232.100",
    "192.168.240.175",
]

data_folder = ["../dataset/train_data", "../dataset/test_data"]

workspace = ['../processed_data/train_data', '../processed_data/test_data']

demo_pcap = black_list[1]

load_layer('tls')


# step1
# 统计一个pcap的基本信息，
# 包括client访问了哪些server
# 每个server访问了几次
# 每一次访问请求的源端口是多少
def summary(ip: str, data_folder=data_folder, workspace=workspace):
    if ip.endswith('.pcap'):
        raise ValueError(ip)
    pcap = Pcap(os.path.join(data_folder, ip + '.pcap'))
    info = {}
    for s, c in pcap.streams:
        if ip in s or ":443" in c:
            s, c = c, s
        logging.debug(f'{c}\t{s}')
        sip = s.split(":")[0]
        cport = int(c.split(":")[1])
        if sip in info:
            info[sip].append(cport)
        else:
            info.update({sip: [cport]})

    Path(os.path.join(os.path.join(workspace, ip))).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(workspace, ip, "info.json"), 'w') as outfile:
        json.dump(info, outfile)


# step2
# 根据上一步的统计信息，
# 根据每一次请求，将数据包分割成多个小的数据包，方便处理
def split(ip, data_folder=data_folder, workspace=workspace):
    info = json.load(open(os.path.join(workspace, ip, "info.json")))
    pcap = Pcap(os.path.join(data_folder, ip + ".pcap"))
    paths = []
    for s, c_ports in info.items():
        Path(os.path.join(workspace, ip, s)).mkdir(parents=True, exist_ok=True)
        for c_port in c_ports:
            logging.debug(f'{ip}: {s} {c_port}')
            path = os.path.join(workspace, ip, s, f'{c_port}.pcap')
            paths.append(path)
            pkts = pcap.filter_by_ip_and_port(s, c_port)
            Pcap.dump_pkts_as_pcap(path, pkts)
            logging.debug(f'{ip}: {c_port} finished')
    return paths


def extractStreamInfo(path):
    pkts = Pcap(path).pkts
    data_path = path[:-5] + ".json"
    try:
        data = json.load(open(data_path))
    except:
        data = {}

    data.update(Pcap.extract(pkts))
    with open(data_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


# 尝试使用递归的方式自动解析提取TLS相关类的Field
# 默认递归深度为 10 
def autoParseClassFields(obj, depth = 20 ):
    if depth <= 0:
        return { "error" : "auto parse depth max!" }

    if obj == None:
        return

    if type(obj) in [int, str, bool]:
        return obj

    if type(obj) == bytes:
        return obj.hex()

    if type(obj) in [list, tuple] :
        return [ autoParseClassFields(i,depth - 1) for i in obj ]


    fields = {}
    if type(obj) == dict:
        for k,v in obj.items():
            if type(k) == str and k[0] != '_':
                fields.update({k: autoParseClassFields(v,depth -1) }) 
        return fields

    fields.update({ "class": obj.__class__.__name__ }) 

    if hasattr(obj,'fields'):
        for k,v in obj.fields.items():
            if type(k) == str and k[0] != '_':
                fields.update({k: autoParseClassFields(v,depth -1) })
    elif hasattr(obj,'__dict__'):
        for k,v in obj.__dict__.items():
            if type(k) == str and k[0] != '_':
                fields.update({k: autoParseClassFields(v,depth -1) }) 
    else:
        return {"error": "cant auto parse this class: {}".format(type(obj))}

    return fields


# 从还原好的raw数据中解析出TLS相关信息
# 返回值：records 和 parsed
# records ：列表形式返回数据
# parsed: 正常情况下等于 len(data), 如果小于len(data) ，说明最后一个 record 的数据不完整


def parseTLSRecords(data):

    records = []
    # print("data len :", len(data))
    parsed = 0

    # tls record 的 header 长度至少为 2 或 5 ，其中 SSLv2 为 2 ， tls 为 5
    # 由于大多数数据为 tls ，在这里向上取 5
    while len(data) - parsed > 5:

        tls = TLS(data[parsed:])

        # 由于 data 中可能存在多个 tls 的 record, 也可能在最后存在一个不完整的 record
        # 且每一个 record 的长度信息，只在 record 的头部记录
        # 因此在 parse 每个 record 的信息之前，因先判断其数据的完整性。
        # 如果不完整，利用 parsed 变量将相关信息返回上层函数，进行后续处理

        if type(tls) == scapy.layers.tls.record.TLS:
            if len(data) - parsed < tls.len + 5:
                return records, parsed
        elif type(tls) == scapy.layers.tls.record_sslv2.SSLv2:
            if len(data) - parsed < tls.len + 2:
                return records, parsed
        else:
            return records, len(data)

        # 利用 scapy.layers.tls 库，对 data 中的信息提取，相关文档地址为：w
        # https://scapy.readthedocs.io/en/latest/api/scapy.layers.tls.html

        record = autoParseClassFields(tls)

        # print(autoParseClassFields(tls))
        # TLS fields :
        # https://scapy.readthedocs.io/en/latest/api/scapy.layers.tls.record.html#id1
        if type(tls) == scapy.layers.tls.record.TLS:
            parsed = parsed + tls.len + 5

        # SSLv2 fields:
        # https://scapy.readthedocs.io/en/latest/api/scapy.layers.tls.record_sslv2.html#id1
        elif type(tls) == scapy.layers.tls.record_sslv2.SSLv2:
            parsed = parsed + tls.len + 2

        records.append(record)

    return records, parsed


# 检查数据包中是否有TLS相关的layer
def checkLayers(p):
    if p.haslayer(TLS):
        return True
    if p.haslayer(SSLv2):
        return True
    if scapy.layers.tls.record._TLSEncryptedContent in p.layers():
        return True

    return False


def extractTLSInfo(path):
    pkts = Pcap(path).pkts

    if pkts[0].getlayer(TCP).flags.flagrepr() != 'S':
        raise Exception("第一个数据包不是syn")

    # 获取 session
    c_ip, c_port, s_ip, s_port = Pcap.get_ip_and_port(pkts[0])

    # 按照当前的处理逻辑，请求和返回的数据包信息，统一整理在变量 records 中
    # 其顺序应该为，每一个 record 最后一个包的出现顺序，而不是第一个包的出现顺序
    # 可能与其数据的请求和返回顺序，不完全一致，如果时间允许，可能需要进一步修改
    records = []
    indexs = []

    type_request = (c_ip, c_port, s_ip, s_port)
    type_response = (s_ip, s_port, c_ip, c_port)

    buff_request = b''
    buff_response = b''

    for i, p in enumerate(pkts):

        if checkLayers(p) == False:
            continue

        tcp = p.getlayer(TCP)

        if Pcap.get_ip_and_port(p) == type_request:
            buff_request = buff_request + raw(tcp.payload)
            # if tcp.flags.flagrepr() == "PA":
            new_records, parsed = parseTLSRecords(buff_request)

            if new_records != []:
                # print(" parsed {} new records from pkts {}".format(len(new_records), i + 1))
                indexs = indexs + [i] * len(new_records)
            records = records + new_records
            buff_request = buff_request[parsed:]

        if Pcap.get_ip_and_port(p) == type_response:
            buff_response = buff_response + raw(tcp.payload)
            # if tcp.flags.flagrepr() == "PA":
            new_records, parsed = parseTLSRecords(buff_response)
            if new_records != []:
                indexs = indexs + [-i] * len(new_records)
                # print(" parsed {} new records from pkts {}".format(len(new_records), i + 1))
            records += new_records
            buff_response = buff_response[parsed:]
    return indexs, records


def loopHosts():
    for p in black_list:
        # summary(p)
        split(p)


def loopServers():
    pass


def extractFromPcap(path:str=None):
    assert path.endswith('.pcap'), f"Not a pcap file: {path}"
    extractStreamInfo(path)

    demo = "demo_out/192.168.7.123/115.231.40.116/1040.pcap" if path is None else path
    out_path = path.replace('.pcap', '.tls.json')
    # logging.debug(out_path)
    # demo = "demo_out/192.168.3.225/106.120.159.182/1034.pcap"
    try:
        indexs, records = extractTLSInfo(demo)
    except (TypeError, KeyError, struct.error, AttributeError) as e:
        logging.warning(f'{path}: {repr(e)}')
        with open(out_path, 'w') as outfile:
            json.dump({"indexs": [], "records": [], 'flag': False}, outfile, indent=4)
        return
    except Exception as e:
        logging.error(f'{path=}')
        traceback.print_exc()
        raise e

    with open(out_path, 'w') as outfile:
        json.dump({"indexs": indexs, "records": records, 'flag': True}, outfile, indent=4)


def walk_test():
    for root, _, files in os.walk('./demo_out'):
        for path in files:
            path = os.path.join(root, path)
            if not path.endswith('pcap'):
                continue
            try:
                extractFromPcap(path)
            except Exception as e:
                print(path)
                traceback.print_exc()
                if 'supported' in repr(e):
                    sys.exit(0)

def test():
    # ip = '192.168.176.245'
    # ip = '192.168.19.170'
    ip = '192.168.36.45'
    split(ip, data_folder[1], workspace[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True)
    args = parser.parse_args()

    if args.action == 'summary':
        for in_path, out_path in zip(data_folder, workspace):
            ips = [i[:-5] for i in os.listdir(in_path) if i.endswith('.pcap') and not i.startswith('.')]
            pool = Pool(processes=24)
            for _ in tqdm(pool.imap_unordered(functools.partial(summary, data_folder=in_path, workspace=out_path), ips), total=len(ips)):
                pass
            # tot = len(ips)
            # fail = 0
            # for ip in tqdm(ips):
            #     try:
            #         summary(ip, in_path, out_path)
            #     except Exception as e:
            #         print(ip)
            #         traceback.print_exc()
            #         sys.exit(0)
            #         fail += 1
            # print(tot, fail)
            # sys.exit(0)
    elif args.action == 'split':
        for in_path, out_path in zip(data_folder, workspace):
            ips = [i[:-5] for i in os.listdir(in_path) if i.endswith('.pcap') and not i.startswith('.')]
            pool = Pool(processes=100)
            for _ in tqdm(pool.imap_unordered(functools.partial(split, data_folder=in_path, workspace=out_path), ips), total=len(ips)):
                pass

            # tot = len(ips)
            # fail = 0
            # for ip in tqdm(ips):
            #     try:
            #         paths = split(ip, in_path, out_path)
            #     except Exception as e:
            #         traceback.print_exc()
            #         print(ip)
            #         sys.exit(0)
            #     for path in paths:
            #         extractStreamInfo(path)
            #         extractTLSInfo(path)
    elif args.action == 'extract':
        paths = []
        for root, folders, files in os.walk('../processed_data'):
            for file in files:
                if not file.endswith('.pcap'):
                    continue
                paths.append(os.path.join(root, file))

        pool = Pool(processes=100)
        for _ in tqdm(pool.imap_unordered(extractFromPcap, paths), total=len(paths)):
            pass
    elif args.action == 'test':
        logging.basicConfig(level=logging.DEBUG, force=True)
        test()
