import sys
import subprocess
from shodan import Shodan
import requests
import time
import tkinter as tk
import requests
from bs4 import BeautifulSoup
from tkinter import scrolledtext
from tkinter import filedialog
from threading import Thread
import socket
from tkinter import ttk
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from tkinter import BooleanVar, Label, Checkbutton, Radiobutton, Button, Entry, Frame
from tkinter import PhotoImage
from PIL import Image, ImageTk
import re

# Replace 'YOUR_SHODAN_API_KEY' with your actual Shodan API key
SHODAN_API_KEY = 'jxLW13mSmgc5PYI3kK1YqUXWvzGZXpbO'
shodan_api = Shodan(SHODAN_API_KEY)


global open_ports_list
open_ports_list = []   

global cveportsgraphdata
cveportsgraphdata = []

global graph_canvas
global graph_figure
graph_canvas = None
graph_figure = None
# Command to open a new Terminal window and run the commands below
def display_open_ports_pie_chart(cveportsgraphdata, results_frame):
    global graph_figure
    global graph_canvas
    
    # Prepare data for the pie chart
    labels = cveportsgraphdata
    sizes = [1 for _ in cveportsgraphdata]  # Equal sizes for each port, adjust as needed
    
    # Create a new figure if it does not exist
    if graph_figure is None:
        graph_figure, ax = plt.subplots()
    else:
        # Clear the existing graph content if the figure already exists
        ax = graph_figure.gca()
        ax.clear()

    # Plot the pie chart
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Handle the canvas
    if graph_canvas is None:
        graph_canvas = FigureCanvasTkAgg(graph_figure, master=results_frame)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack()
    else:
        graph_canvas.draw()

def show_open_ports_pie_chart():
    global cveportsgraphdata
    global graph_canvas
    global graph_figure
    
    if not cveportsgraphdata:  # Check if the list is empty
        print("No CVE-related open ports data available. Please perform a scan first.")
        return
    
    # Check if the graph is already displayed
    if graph_canvas is not None:
        # Destroy the graph canvas and figure to close the graph
        graph_canvas.get_tk_widget().destroy()
        graph_canvas = None
        graph_figure = None
    else:
        # Display the graph if it's not currently shown
        display_open_ports_pie_chart(cveportsgraphdata, resultsframe)


        
def runmetasploit():
    if runmetasploitterminal.lower() == ("yes"):
        commands = '''
        /opt/metasploit-framework/bin/msfconsole
        msfrpcd -P password -S -f
        '''
        command = f'osascript -e \'tell app "Terminal" to do script "{commands}"\''
        subprocess.run(["/bin/bash", "-c", command])
    elif runmetasploitterminal.lower() == ("no"):
        dontwannarunmeta = "Metasploit running canceled"
        return print(dontwannarunmeta)

runmetasploitterminal = input("Do you want to run Metasploit yes or no: ")
runmetasploit()

#Metasploit Console
MSF_RPC_HOST = "127.0.0.1"
MSF_RPC_PORT = "55553"
MSF_RPC_URL = f"http://{MSF_RPC_HOST}:{MSF_RPC_PORT}/api/"
MSF_RPC_USER = "msf"  # Default user
MSF_RPC_PASSWORD = "password"  # The password you set when starting the RPC server

def authenticate_rpc():
    try:
        response = requests.post(MSF_RPC_URL + 'auth', json={"username": MSF_RPC_USER, "password": MSF_RPC_PASSWORD})
        if response.status_code == 200:
            return response.json()['token']
        else:
            print("Failed to authenticate with Metasploit RPC server.")
            return None
    except Exception as e:
        print(f"Error during authentication with Metasploit RPC server: {e}")
        return None

def execute_rpc_command(token, method, params=[]):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    data = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': method,
        'params': params
    }
    try:
        response = requests.post(MSF_RPC_URL + 'execute', headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"RPC command execution failed: {response.text}")
            return None
    except Exception as e:
        print(f"Error executing RPC command: {e}")
        return None
    
    

    
def scan_ip_for_open_ports(ip, options, timing='T4'):
    cmd = ['nmap', ip]

    #options based on the BooleanVars/checkboxes
    if options['fast_scan'].get():
        cmd.append('-F')
    if options['show_open'].get():
        cmd.append('--open')
    if options['version_detection'].get():
        cmd.append('-sV')
    if options['os_detection'].get():
        cmd.append('-O')
    if options['script_scan'].get():
        cmd.append('-sC')
    if options['aggressive_scan'].get():
        cmd.append('-A')
    if options['no_ping'].get():
        cmd.append('-Pn')
    if options['stealth_scan'].get():
        cmd.append('-sS')
    if options['udp_scan'].get():
        cmd.append('-sU')
        
        
    if options['vulnerability_scan'].get():
        cmd.append('--script=vulners')
        
        
    cmd.append(f'-{timing.get()}')  #timing default is T4

        
    try:
        print(f"Running command: {' '.join(cmd)}")
        print(cmd)
        result = subprocess.check_output(cmd, universal_newlines=True)
        return parse_nmap_vuln_output(result)
    except Exception as e:
        print(f"Error scanning {ip} for open ports: {e}")
        return []
    
#old method of getting nmap outputs
""" def parse_nmap_vuln_output(nmap_output):
    open_ports = []
    nmap_vulnerabilities = {}
    for line in nmap_output.split('\n'):
        if '/tcp' in line and 'open' in line:
            port = line.split('/')[0].strip()
            open_ports.append(port)
        if 'CVE' in line:
            parts = line.split('|')
            for part in parts:
                if 'CVE-' in part:
                    cve = part.strip().split()[0]
                    if port not in nmap_vulnerabilities:
                        nmap_vulnerabilities[port] = []
                    nmap_vulnerabilities[port].append(cve)
    return open_ports, nmap_vulnerabilities """

def parse_nmap_vuln_output(nmap_output):
    open_ports = []
    nmap_vulnerabilities = {}
    cve_pattern = re.compile(r"CVE-\d{4}-\d{4,7}")  #Pattern to match typical CVE IDs
    for line in nmap_output.split('\n'):
        if '/tcp' in line and 'open' in line:
            port = line.split('/')[0].strip()
            open_ports.append(port)
        if 'CVE' in line:
            parts = line.split('|')
            for part in parts:
                match = cve_pattern.search(part)
                if match:
                    cve = match.group()
                    if port not in nmap_vulnerabilities:
                        nmap_vulnerabilities[port] = []
                    nmap_vulnerabilities[port].append(cve)
    return open_ports, nmap_vulnerabilities


def get_vulnerabilities_from_shodan(ip, ports):
    vulnerabilities = {}
    try:
        host_info = shodan_api.host(ip)
        for port in ports:
            service_info = host_info.get('data', [])
            for service in service_info:
                if service.get('port') == int(port):
                    vulns = service.get('vulns', [])
                    vulnerabilities[port] = vulns
        return vulnerabilities
    except Exception as e:
        print(f"Error retrieving vulnerabilities from Shodan for {ip}: {e}")
        return {}





def search_exploitdb(all_cve_ids):
    exploits = {}
    for cve_id in all_cve_ids:
        try:
            # Execute the searchsploit command with the CVE number
            result = subprocess.run(["searchsploit", "--json", cve_id], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse the output and store the URL or the entire output as needed
                exploits[cve_id] = result.stdout
                results_text.insert(tk.INSERT, f"Results for {cve_id}:")
                results_text.insert(tk.INSERT, result.stdout)  # Print the results for this particular CVE
            else:
                error_message = f"searchsploit did not find any results for {cve_id} or failed to run"
                exploits[cve_id] = error_message
                results_text.insert(tk.INSERT, error_message)
        except Exception as e:
            error_message = f"Error searching ExploitDB for {cve_id}: {e}"
            exploits[cve_id] = error_message
            results_text.insert(tk.INSERT, error_message)
    return exploits




    
def select_file():
    file_path = filedialog.askopenfilename()
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)    
    
     
def analyze_file_gui():
    global open_ports_list
    
    global cveportsgraphdata

    ip_file = file_entry.get()
    if not ip_file:
        results_text.insert(tk.INSERT, "Please select a file.\n")
        return
    
    try:
        with open(ip_file, 'r') as file:
            for ip in file:
                ip = ip.strip()
                IP = ip
                
                progress_bar.start(2)

                
                if IP:
                    results_text.insert(tk.INSERT, f"Analyzing IP: {IP}\n---------------------------------\n")
                    
                    open_ports_list, nmap_vulnerabilities = scan_ip_for_open_ports(IP, options, timing)
                    results_text.insert(tk.INSERT, f"Open ports:\n")
        
                    for port in open_ports_list:
                        results_text.insert(tk.INSERT, f" - Port {port}\n")
                    
                    results_text.insert(tk.INSERT, "\n---------------------------------------------------------------------\n")

                        
                    shodan_vulnerabilities = get_vulnerabilities_from_shodan(IP, open_ports_list)
                    
                    all_vulnerabilities = {}
                    ports_with_cves = []

                    # Merge Shodan and Nmap vulnerabilities
                    for port in open_ports_list:
                        merged_vulns = set(shodan_vulnerabilities.get(port, [])) | set(nmap_vulnerabilities.get(port, []))
                        all_vulnerabilities[port] = list(merged_vulns)
                        
                        if merged_vulns:
                            ports_with_cves.append(port)
                            
                    cveportsgraphdata = ports_with_cves
                    
                    all_cve_ids = [] 
                    for port, vulns in all_vulnerabilities.items():
                        results_text.insert(tk.INSERT, f"\nVulnerabilities for Port {port}:\n")
                        if vulns:
                            for vuln in vulns:
                                results_text.insert(tk.INSERT, f" - {vuln}\n")
                                all_cve_ids.append(vuln)
                                
                                print(vulns)#debugging
                                
                        else:
                            results_text.insert(tk.INSERT, " - None found\n")
                        
                    results_text.insert(tk.INSERT, "\n---------------------------------------------------------------------\n")
                    results_text.insert(tk.INSERT, "\n")
                    search_exploitdb(all_cve_ids)    
                        
    except FileNotFoundError:
        results_text.insert(tk.INSERT, f"File {ip_file} not found.\n")
    except Exception as e:
        results_text.insert(tk.INSERT, f"An error occurred: {e}\n")
    finally:
        results_text.see(tk.END) #newline see end
        progress_bar.stop()
            
    if all_cve_ids:  #process if there are CVE IDs collected
        threadedprocess_cve_ids(all_cve_ids)
         
         
            
def analyze_threaded():
    # Run the analysis in a separate thread to prevent GUI freezing
    analysis_thread = Thread(target=analyze_file_gui)
    analysis_thread.start()
    
    
    
    
#global open_ports_list
#open_ports_list = []     
    
def analyze_ip_gui():
    #below global var new
    global open_ports_list
    
    global cveportsgraphdata

    iptarget = Ip_entry.get()
    if not iptarget:
        results_text.insert(tk.INSERT, "Please insert an IP target to scan.\n")
        return
    progress_bar.start(2)
    IP = iptarget
    results_text.insert(tk.INSERT, f"Analyzing IP: {IP}\n---------------------------------\n")

    try:
        
        open_ports_list, nmap_vulnerabilities = scan_ip_for_open_ports(iptarget, options, timing)

        results_text.insert(tk.INSERT, f"Open ports:\n")
        
        for port in open_ports_list:
            results_text.insert(tk.INSERT, f" - Port {port}\n")
        
        
        
        results_text.insert(tk.INSERT, "\n---------------------------------------------------------------------\n")

        shodan_vulnerabilities = get_vulnerabilities_from_shodan(iptarget, open_ports_list)
        
        all_vulnerabilities = {}
        ports_with_cves = []
        
        # Merge Shodan and Nmap vulnerabilities
        for port in open_ports_list:
            merged_vulns = set(shodan_vulnerabilities.get(port, [])) | set(nmap_vulnerabilities.get(port, []))
            all_vulnerabilities[port] = list(merged_vulns)
            
            if merged_vulns:
                ports_with_cves.append(port)
                            
        cveportsgraphdata = ports_with_cves
        
        
        
        all_cve_ids = [] 
        for port, vulns in all_vulnerabilities.items():
            results_text.insert(tk.INSERT, f"\nVulnerabilities for Port {port}:\n")
            if vulns:
                for vuln in vulns:
                    results_text.insert(tk.INSERT, f" - {vuln}\n")
                    all_cve_ids.append(vuln)
                    
            else:
                results_text.insert(tk.INSERT, " - None found\n")
                
        results_text.insert(tk.INSERT, "\n---------------------------------------------------------------------\n")
        results_text.insert(tk.INSERT, "\n")
        search_exploitdb(all_cve_ids)    
            
    except Exception as e:
        results_text.insert(tk.INSERT, f"An error occurred during analysis: {e}\n")
    finally:
        results_text.see(tk.END) #newline see end
        progress_bar.stop()
        
    if all_cve_ids:  #process if there are CVE IDs collected
        threadedprocess_cve_ids(all_cve_ids)


def display_exploit_details(cve_id):
    api_url = f"https://cve.circl.lu/api/cve/{cve_id}"
    response = requests.get(api_url)
    if response.status_code == 200:
        cve_details = response.json()
    else:
        results_text.insert(tk.INSERT, "\nFailed to retrieve CVE details. Please check the CVE ID and your internet connection.\n")
        return None
    
    if cve_details is None:
        results_text.insert(tk.INSERT, "No details available to display.\n")
        return
    
    #results_text.insert(tk.INSERT, "\n")
    results_text.insert(tk.INSERT, "\n---------------------------------------------------------------------\n")
    results_text.insert(tk.INSERT, f"\nDetails for CVE: {cve_details.get('id')}\n")
    results_text.insert(tk.INSERT, "\n")
    results_text.insert(tk.INSERT, f"\nSummary: {cve_details.get('summary')}\n")
    results_text.insert(tk.INSERT, "\n---------------------------------------------------------------------\n")
          
    if 'references' in cve_details and cve_details['references']:
        results_text.insert(tk.INSERT, (""))
        results_text.insert(tk.INSERT, ("\nReferences:\n"))
        for ref in cve_details['references'][:5]: #num is amount refrences to print
            results_text.insert(tk.INSERT, f"\n{ref}\n") #could just be (ref)
    else:
        results_text.insert(tk.INSERT, ("\nNo references available.\n"))
        
    results_text.insert(tk.INSERT, "\n---------------------------------------------------------------------\n")
    results_text.insert(tk.INSERT, (f"\nSearching GitHub for potential exploits for {cve_id}...\n"))
    search_url = f"https://github.com/search?q={cve_id}+exploit&type=Repositories"
    results_text.insert(tk.INSERT, (f"\nCheck out GitHub search results at: {search_url}\n"))
    query = f"https://www.google.com/search?q={cve_id}+exploit"
    results_text.insert(tk.INSERT, (f"\nTo search for more exploit details on Google use this query: {query}\n"))
    results_text.insert(tk.INSERT, "\n---------------------------------------------------------------------\n")

def fetch_and_display_cve_details(cve_id):
    display_exploit_details(cve_id)
        
        
        
def threadedprocess_cve_ids(cve_ids):
    for cve_id in cve_ids:
        print(f"Processing {cve_id}...")
        thread = Thread(target=fetch_and_display_cve_details, args=(cve_id,))
        thread.start()



def analyze_threadedIPtarget():
    analysis_thread = Thread(target=analyze_ip_gui)
    analysis_thread.start()
    findipcam()

def ipinfo():
    ipinfotarget = Ip_entry.get()
    api = "http://ip-api.com/json/"
    
    data = requests.get(api + ipinfotarget).json()
    if not ipinfotarget:
        results_text.insert(tk.INSERT, "Please insert a IP to get info on.\n")
        return
    if ipinfotarget:
        output = [
                "[Victim]: " + str(data.get('query', 'Unknown')),
                "[ISP]: " + str(data.get('isp', 'Unknown')),
                "[Organisation]: " + str(data.get('org', 'Unknown')),
                "[City]: " + str(data.get('city', 'Unknown')),
                "[Region]: " + str(data.get('region', 'Unknown')),
                "[Longitude]: " + str(data.get('lon', 'Unknown')),
                "[Latitude]: " + str(data.get('lat', 'Unknown')),
                "[Time zone]: " + str(data.get('timezone', 'Unknown')),
                "[Zip code]: " + str(data.get('zip', 'Unknown')),
        ]

        resultsipinfo = "\n".join(output)
        results_text.insert(tk.INSERT, resultsipinfo + "\n")    
        results_text.insert(tk.INSERT, "\n")
        results_text.insert(tk.INSERT, "\n")

    
        
def threadedipinfo():
    # Run the analysis in a separate thread to prevent GUI freezing
    analysis_thread = Thread(target=ipinfo)
    analysis_thread.start()   
    

def get_ip_website():
    domain = website_entry.get()
    try:
        ip_address = socket.gethostbyname(domain)
        return results_text.insert(tk.INSERT, ip_address + "\n")
    except socket.gaierror as e:
        return f"Error getting IP for {domain}: {e}"

def threadedwebsiteipinfo():
    analysis_thread = Thread(target=get_ip_website)
    analysis_thread.start()  


def findipcam():
    ipcamip = Ip_entry.get()
    #webbrowser.open(f"http://{ipcamip}:80")
    results_text.insert(tk.INSERT, ("If an IP Cam exists it may be at this link") + "\n")
    ipcaminfolink1 = f"http://{ipcamip}:80"
    ipcaminfolink2 = f"http://{ipcamip}:443"
    ipcaminfolink3 = f"http://{ipcamip}:554"
    results_text.insert(tk.INSERT, ipcaminfolink1 + "\n")
    results_text.insert(tk.INSERT, ipcaminfolink2 + "\n")
    results_text.insert(tk.INSERT, ipcaminfolink3 + "\n")
    return results_text.insert(tk.INSERT, "\n")

    
    
def clear_view():
    #Insert a large number of newlines to 'push' older content out of immediate view
    results_text.insert(tk.END, "\n" * 70)
    #Automatically scroll to the bottom of the widget
    results_text.see(tk.END)

def exitprogram():
    sys.exit()
        
        
#the GUI
root = tk.Tk()
root.title("SpiderCrawler")

#overall size of program
root.geometry("1050x950")  # 800x600Set initial window size

#window resizable
root.resizable(True, True)

ipframe = tk.Frame(root, padx=10, pady=10)
ipframe.pack(side=tk.TOP)
ipframe.pack(side=tk.LEFT)

Ip_name = tk.Label(ipframe, text="Ip Target:")
Ip_name.pack(side=tk.LEFT, padx=10, pady=10)

Ip_entry = tk.Entry(ipframe, width=12)
Ip_entry.pack(side=tk.LEFT, padx=10, pady=10)



options = {
    'fast_scan': BooleanVar(value=False),
    'show_open': BooleanVar(value=False),
    'version_detection': BooleanVar(value=False),
    'os_detection': BooleanVar(value=False),
    'script_scan': BooleanVar(value=False),
    'aggressive_scan': BooleanVar(value=False),
    'no_ping': BooleanVar(value=False),
    'stealth_scan': BooleanVar(value=False),
    'udp_scan': BooleanVar(value=False),
    'vulnerability_scan': BooleanVar(value=False)

}
timing = tk.StringVar(value="T4")  # Default timing is T4



#Checkboxes for different scanning options
Checkbutton(ipframe, text="Fast Scan (Most Common Ports): -F ", var=options['fast_scan']).pack(anchor='w')
Checkbutton(ipframe, text="Show Only Open Ports: --open ", var=options['show_open']).pack(anchor='w')
Checkbutton(ipframe, text="Version Detection: -sV ", var=options['version_detection']).pack(anchor='w')
Checkbutton(ipframe, text="OS Detection: -O ", var=options['os_detection']).pack(anchor='w')
Checkbutton(ipframe, text="Script Scan (Default Scripts): -sC ", var=options['script_scan']).pack(anchor='w')
Checkbutton(ipframe, text="Aggressive Scan: -A ", var=options['aggressive_scan']).pack(anchor='w')
Checkbutton(ipframe, text="No Ping (Skip Discovery): -Pn ", var=options['no_ping']).pack(anchor='w')
Checkbutton(ipframe, text="Stealth SYN Scan: -sS ", var=options['stealth_scan']).pack(anchor='w')
Checkbutton(ipframe, text="UDP Scan: -sU ", var=options['udp_scan']).pack(anchor='w')
Checkbutton(ipframe, text="Vulnerability Scan: --script vulners ", var=options['vulnerability_scan']).pack(anchor='w')

#Timing options
Label(ipframe, text="Timing Template:").pack(anchor='w')
for t in ['T2', 'T3', 'T4', 'T5']:
    Radiobutton(ipframe, text=t, value=t, variable=timing).pack(anchor='w')



ipanalyze_button = tk.Button(ipframe, text="IPAnalyze", command=analyze_threadedIPtarget)
ipanalyze_button.pack(side=tk.LEFT)
ipanalyze_button.pack(side=tk.BOTTOM)


ipinfo_button = tk.Button(ipframe, text="IPinfo", command=threadedipinfo)
ipinfo_button.pack(side=tk.LEFT)
ipinfo_button.pack(side=tk.TOP)


show_graph_button = tk.Button(ipframe, text="Show Ports Graph", command=show_open_ports_pie_chart)
show_graph_button.pack(side=tk.LEFT)
show_graph_button.pack(side=tk.BOTTOM)




websiteframe = tk.Frame(root, padx=10, pady=10)
websiteframe.pack(side=tk.TOP)

website_name = tk.Label(websiteframe, text="Domain:")
website_name.pack(side=tk.LEFT)

website_entry = tk.Entry(websiteframe, width=25)
website_entry.pack(side=tk.LEFT)

website_button = tk.Button(websiteframe, text="Find Domain IP", command=threadedwebsiteipinfo)
website_button.pack(side=tk.RIGHT)




fileframe = tk.Frame(root, padx=10, pady=10)
fileframe.pack(side=tk.TOP)

file_label = tk.Label(fileframe, text="File Path:")
file_label.pack(side=tk.LEFT)

file_entry = tk.Entry(fileframe, width=25)
file_entry.pack(side=tk.LEFT)

browse_button = tk.Button(fileframe, text="Browse", command=select_file)
browse_button.pack(side=tk.LEFT)

analyze_button = tk.Button(fileframe, text="IPFileAnalyze", command=analyze_threaded)
analyze_button.pack(side=tk.BOTTOM)





resultsframe = tk.Frame(root, padx=10, pady=10)
resultsframe.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

#initialize progress bar
progress_bar = ttk.Progressbar(resultsframe, orient='horizontal', mode='indeterminate', length=280)
progress_bar.pack(side=tk.TOP, pady=10)

clear_button = tk.Button(resultsframe, text="Clear View", command=clear_view)
clear_button.pack(side=tk.LEFT)
clear_button.pack(side=tk.TOP)

exit_button = tk.Button(resultsframe, text="Exit/Cancel", command=exitprogram)
exit_button.pack(side=tk.RIGHT)
exit_button.pack(side=tk.TOP)

results_text = scrolledtext.ScrolledText(resultsframe, width=70, height=20)
results_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
results_text.pack(side=tk.BOTTOM)


root.mainloop()