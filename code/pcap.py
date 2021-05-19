#! /usr/bin/python3
# -*- coding:UTF-8 -*-
import logging
import argparse
import os
import json
import string
import re
from scapy.all import *
from builtins import bytes


class Pcap:
    def __init__(self, filename, filter="ip and tcp"):
        self.filename = filename
        self.pkts = sniff(offline=filename, filter=filter, verbose=0, quiet=True)
        self.sessions = self.pkts.sessions()
        self.streams = self.get_streams()

    def get_streams(self):
        """stream对应的是往返ip和端口的所有数据包,返回数组：[('127.0.0.1:4444', '127.0.0.1:57532')]"""
        return list(set(tuple(sorted(session.replace("TCP ", "").split(" > "))) for session in self.sessions.keys()))

    def search(self, data):
        """返回在tcp层数据存在data的pkt列表，其中data是bytes格式"""
        return [p for p in self.pkts if data in raw(p.getlayer(TCP).payload)]

    def regex(self, pattern):
        return [p for p in self.pkts if re.search(pattern, raw(p.getlayer(TCP).payload))]

    def filter_by_BPF(self, bpf, pkts=None):
        """使用伯克利包过滤(Berkeley Packet Filter)规则,过滤数据包"""
        pkts = pkts or self.pkts
        return sniff(offline=pkts, filter=bpf, verbose=0, quiet=True)

    def filter_by_ip(self, ip, pkts=None):
        bpf = "ip and host {}".format(ip)
        return self.filter_by_BPF(bpf, pkts)

    def filter_by_port(self, port, pkts=None):
        bpf = "ip and port {}".format(port)
        return self.filter_by_BPF(bpf, pkts)

    def filter_by_ip_and_port(self, ip, port, pkts=None):
        bpf = "ip and host {} and port {}".format(ip, port)
        return self.filter_by_BPF(bpf, pkts)

    def follow_tcp_stream(self, pkt):
        """从任意pkt，返回其所在的tcp流的所有pkts"""
        ip = pkt.getlayer(IP)
        tcp = pkt.getlayer(TCP)
        pkts = self.filter_by_ip_and_port(ip.src, tcp.sport)
        return self.filter_by_ip_and_port(ip.dst, tcp.dport, pkts)

    def payload_to_ascii(self, payload):
        if payload != None:
            return re.sub(b'[^\x1f-\x7f]', b'.', payload).decode()

    def payload_to_carray(self, payload):
        return ",".join("{:#04x}".format(b) for b in bytes(payload))

    def dump_stream_pkts(self, filename, pkts):

        if pkts[0].getlayer(TCP).flags.flagrepr() != 'S':
            raise Exception("第一个数据包不是syn")

        src, sport, dsc, dport = Pcap.get_ip_and_port(pkts[0])

        type_request = (src, sport, dsc, dport)
        type_response = (dsc, dport, src, sport)

        count_request = 0
        count_response = 0

        res = []

        for i, pkt in enumerate(pkts):
            payload = raw(pkt.getlayer(TCP).payload)
            if payload == b'':
                continue

            if Pcap.get_ip_and_port(pkt) == type_request:
                name = "peer{}_{}".format(0, count_request)
                count_request += 1
            elif Pcap.get_ip_and_port(pkt) == type_response:
                name = "peer{}_{}".format(1, count_response)
                count_response += 1
            else:
                raise Exception("存在不是")

            res.append({
                "name": name,
                "time": pkt.time,
                "index": i + 1,
                "ascii": self.payload_to_ascii(payload),
                "carray": self.payload_to_carray(payload)
            })

        with open(filename, 'w') as outfile:
            json.dump(res, outfile)

    @staticmethod
    def dump_pkts_as_pcap(filename, pkts):
        wrpcap(filename, pkts)

    @staticmethod
    def get_ip_and_port(pkt):
        ip = pkt.getlayer(IP)
        tcp = pkt.getlayer(TCP)
        return ip.src, tcp.sport, ip.dst, tcp.dport

    @staticmethod
    def extract(pkts):
        """从流数据中提取基本信息"""
        if pkts[0].getlayer(TCP).flags.flagrepr() != 'S':
            raise Exception("第一个数据包不是syn")

        src, sport, dsc, dport = Pcap.get_ip_and_port(pkts[0])

        type_request = (src, sport, dsc, dport)
        type_response = (dsc, dport, src, sport)

        lenth_vector = []
        time_vector = []
        for pkt in pkts:
            time_vector.append(float(pkt.time))
            payload = raw(pkt.getlayer(TCP).payload)
            if Pcap.get_ip_and_port(pkt) == type_request:
                lenth_vector.append(len(payload))
            elif Pcap.get_ip_and_port(pkt) == type_response:
                lenth_vector.append(-len(payload))
            else:
                logging.error(f'{Pcap.get_ip_and_port(pkt)}, {type_request=}, {type_response=}')
                ls(pkt)
                raise Exception("存在不是")
        
        return {"time":time_vector, "lenth":lenth_vector }


def test():
    pcap = Pcap("../dataset/train_data/192.168.0.10.pcap")
    # # print(pcap.streams)
    # ps = pcap.regex(b"suffarring")
    e = Pcap.extract(pcap.pkts)
    print(e['time'])
    # pkts = pcap.follow_tcp_stream(ps[0])
    # print(pcap.dump_stream_pkts("test.json", pkts))
    # print(list(pcap.sessions.keys()))
    # pcap.summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("filename")
    parser.add_argument("action", choices=["list", "search", "dump"])
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('-i', '--inputfile', nargs='?', default=None)
    # parser.add_argument('-o', '--outputfile', nargs='?', default=None)
    # parser.add_argument('--save', type=int, default=3, help='0 no ; 1 carrsys ; 2 ascii and carrsys ; 3 tcpstreams.pcap ascii and carrsys')
    parser.add_argument('-f', '--filter', nargs='?')
    parser.add_argument('-i', '--ip', nargs='?')
    parser.add_argument('-p', '--port', nargs='?')
    parser.add_argument('-s', '--search', nargs='?')
    # parser.add_argument('-pl', '--protocol', nargs='?', default="tcp")
    # args = parser.parse_args()

    # handler = Pcap(args.inputfile)
    # handler.handle(infilename=args.inputfile, outfilename=args.outputfile, filter_string=args.filter, ip=args.ip, port=args.port, query_string=args.search, save=args.save, protocol=args.protocol)
    test()
