import sys
import argparse
import datetime
import pyshark
import os
from netaddr import IPNetwork, IPAddress, IPSet

fileOutput = 'data/file_out.dat'

def pktHandler(timestamp,srcIP,dstIP,lengthIP,sampDelta):
    global scnets
    global ssnets
    global npkts
    global T0
    global outc
    global last_ks
    global fileOutput
    
    with open(fileOutput, 'a') as file_obj:
        if (IPAddress(srcIP) in scnets) or (IPAddress(dstIP) in scnets) or (IPAddress(srcIP) in ssnets and IPAddress(dstIP) in scnets):
            if npkts==0:
                T0=float(timestamp)
                last_ks=0
                
            ks=int((float(timestamp)-T0)/sampDelta)
            
            if ks>last_ks:
                file_obj.write('{} {} {} {} {}\n'.format(last_ks,*outc))
                outc=[0,0,0,0]  
                
            if ks>last_ks+1:
                for j in range(last_ks+1,ks):
                    file_obj.write('{} {} {} {} {}\n'.format(j,*outc))
                    
            
            if IPAddress(srcIP) in scnets: #Upload
                outc[0]=outc[0]+1
                outc[1]=outc[1]+int(lengthIP)

            if IPAddress(dstIP) in scnets: #Download
                outc[2]=outc[2]+1
                outc[3]=outc[3]+int(lengthIP)

            last_ks=ks
            npkts=npkts+1

    #file_obj.close()


def main():

    IPv4 = ['192.168.1.83', '10.0.2.15']
    IPv6 = ['2a05:d018:76c:b684:8e48:47c9:84aa:b34d', '2001:818:e4f2:5::10', '2001:818:eb6f:8f00:a8f2:119:4fa7:c82e']
    YOUTUBE = ['172.217.17.14']
    BROWSING = ['69.171.250.35', '104.244.42.130', '142.250.184.164', '193.137.20.123', '151.101.133.140']
    SPOTIFY = ['2600:1901:1:64a::', '2600:1901:1:c36::']
    NETFLIX = [] # tcp.port == 35856
    MINING = [] # tcp.port == 3380 
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input file, capture .pcap')
    parser.add_argument('-o', '--output', nargs='?',required=True, help='output file')
    parser.add_argument('-si', '--sampleinterval', nargs='?',type=int, help='sample interval', default=1)
    
    args=parser.parse_args()
    
    cnets=[]
    #for n in args.cnet:
    for n in IPv6:
        try:
            nn=IPNetwork(n)
            cnets.append(nn)
        except:
            print('{} is not a network prefix'.format(n))
    #print(cnets)
    if len(cnets)==0:
        print("No valid client network prefixes.")
        sys.exit()
    global scnets
    scnets=IPSet(cnets)

    snets=[]
    for n in SPOTIFY:
        try:
            nn=IPNetwork(n)
            snets.append(nn)
        except:
            print('{} is not a network prefix'.format(n))
    #print(snets)
    if len(snets)==0:
        print("No valid service network prefixes.")
        #sys.exit()
        
    global ssnets
    ssnets=IPSet(snets)

    global fileOutput
    fileInput=args.input
    fileOutput=args.output
        
    global npkts
    global T0
    global outc
    global last_ks

    npkts=0
    outc=[0,0,0,0]
    sampDelta= args.sampleinterval

    try:
        capture = pyshark.FileCapture(fileInput, display_filter='tcp')
        for pkt in capture:
            if 'ipv6' in [l.layer_name for l in pkt.layers]:
                timestamp,srcIP,dstIP,lengthIP=pkt.sniff_timestamp,pkt.ipv6.src,pkt.ipv6.dst,pkt.ipv6.plen
                pktHandler(timestamp,srcIP,dstIP,lengthIP,sampDelta)
            else:
                timestamp,srcIP,dstIP,lengthIP=pkt.sniff_timestamp,pkt.ip.src,pkt.ip.dst,pkt.ip.len
                pktHandler(timestamp,srcIP,dstIP,lengthIP,sampDelta)

    except Exception as e:
        print(e)
        print('\nCapture reading interrupted')
    
    

if __name__ == '__main__':
    main()

