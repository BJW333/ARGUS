import sys
import subprocess
from shodan import Shodan
import requests
import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog
from threading import Thread
import socket
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import BooleanVar, Label, Checkbutton, Radiobutton
import re
import queue

SHODAN_API_KEY = 'jxLW13mSmgc5PYI3kK1YqUXWvzGZXpbO'
shodan_api = Shodan(SHODAN_API_KEY)

open_ports_list = []
cveportsgraphdata = []
graph_canvas = None
graph_figure = None
gui_queue = queue.Queue()

def display_open_ports_pie_chart(cveportsgraphdata, results_frame):
    global graph_figure
    global graph_canvas

    labels = cveportsgraphdata
    sizes = [1 for _ in cveportsgraphdata]  #equal sizes for each port

    #create a new figure if it does not exist
    if graph_figure is None:
        graph_figure, ax = plt.subplots()
    else:
        #clear the existing graph content if the figure already exists
        ax = graph_figure.gca()
        ax.clear()

    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  #ensures that pie is drawn as a circle.

    #handle the canvas
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

    if not cveportsgraphdata:
        print("No CVE-related open ports data available. Please perform a scan first.")
        return

    #check if the graph is already displayed
    if graph_canvas is not None:
        #close the graph
        graph_canvas.get_tk_widget().destroy()
        graph_canvas = None
        graph_figure = None
    else:
        #display the graph
        display_open_ports_pie_chart(cveportsgraphdata, resultsframe)

def scan_ip_for_open_ports(ip, options, timing='T4'):
    cmd = ['nmap', ip]

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

    cmd.append(f'-{timing}')  #timing parameter

    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.check_output(cmd, universal_newlines=True)
        return parse_nmap_vuln_output(result)
    except Exception as e:
        print(f"Error scanning {ip} for open ports: {e}")
        return [], {}

def parse_nmap_vuln_output(nmap_output):
    open_ports = []
    nmap_vulnerabilities = {}
    cve_pattern = re.compile(r"CVE-\d{4}-\d{4,7}")  #pattern to match typical CVE IDs
    port = None
    for line in nmap_output.split('\n'):
        if '/tcp' in line and 'open' in line:
            port = line.split('/')[0].strip()
            open_ports.append(port)
        if 'CVE' in line and port:
            matches = cve_pattern.findall(line)
            if matches:
                if port not in nmap_vulnerabilities:
                    nmap_vulnerabilities[port] = []
                nmap_vulnerabilities[port].extend(matches)
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
            #searchsploit command with the CVE number
            result = subprocess.run(["searchsploit", "--json", cve_id], capture_output=True, text=True)
            if result.returncode == 0:
                #parse the output and store the URL or the entire output as needed
                exploits[cve_id] = result.stdout
                output = f"Results for {cve_id}:\n{result.stdout}\n"
            else:
                error_message = f"searchsploit did not find any results for {cve_id} or failed to run\n"
                exploits[cve_id] = error_message
                output = error_message
        except Exception as e:
            error_message = f"Error searching ExploitDB for {cve_id}: {e}\n"
            exploits[cve_id] = error_message
            output = error_message
        gui_queue.put(output)
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
        gui_queue.put("Please select a file.\n")
        return

    all_cve_ids = []

    try:
        with open(ip_file, 'r') as file:
            for ip in file:
                ip = ip.strip()
                IP = ip

                progress_bar.start(2)

                if IP:
                    output = f"Analyzing IP: {IP}\n---------------------------------\n"

                    open_ports_list, nmap_vulnerabilities = scan_ip_for_open_ports(IP, options, timing.get())
                    output += f"Open ports:\n"

                    for port in open_ports_list:
                        output += f" - Port {port}\n"

                    output += "\n---------------------------------------------------------------------\n"

                    shodan_vulnerabilities = get_vulnerabilities_from_shodan(IP, open_ports_list)

                    all_vulnerabilities = {}
                    ports_with_cves = []

                    #Shodan and Nmap vulnerabilities
                    for port in open_ports_list:
                        merged_vulns = set(shodan_vulnerabilities.get(port, [])) | set(nmap_vulnerabilities.get(port, []))
                        all_vulnerabilities[port] = list(merged_vulns)

                        if merged_vulns:
                            ports_with_cves.append(port)

                    cveportsgraphdata = ports_with_cves

                    for port, vulns in all_vulnerabilities.items():
                        output += f"\nVulnerabilities for Port {port}:\n"
                        if vulns:
                            for vuln in vulns:
                                output += f" - {vuln}\n"
                                all_cve_ids.append(vuln)
                        else:
                            output += " - None found\n"

                    output += "\n---------------------------------------------------------------------\n\n"
                    gui_queue.put(output)
                    search_exploitdb(all_cve_ids)
    except FileNotFoundError:
        gui_queue.put(f"File {ip_file} not found.\n")
    except Exception as e:
        gui_queue.put(f"An error occurred: {e}\n")
    finally:
        progress_bar.stop()

    if all_cve_ids:
        threadedprocess_cve_ids(all_cve_ids)

def analyze_threaded():
    analysis_thread = Thread(target=analyze_file_gui)
    analysis_thread.start()

def analyze_ip_gui():
    global open_ports_list
    global cveportsgraphdata

    iptarget = Ip_entry.get()
    if not iptarget:
        gui_queue.put("Please insert an IP target to scan.\n")
        return

    progress_bar.start(2)
    IP = iptarget
    output = f"Analyzing IP: {IP}\n---------------------------------\n"

    all_cve_ids = []

    try:
        open_ports_list, nmap_vulnerabilities = scan_ip_for_open_ports(iptarget, options, timing.get())

        output += f"Open ports:\n"

        for port in open_ports_list:
            output += f" - Port {port}\n"

        output += "\n---------------------------------------------------------------------\n"

        shodan_vulnerabilities = get_vulnerabilities_from_shodan(iptarget, open_ports_list)

        all_vulnerabilities = {}
        ports_with_cves = []

        #Shodan and Nmap vulnerabilities
        for port in open_ports_list:
            merged_vulns = set(shodan_vulnerabilities.get(port, [])) | set(nmap_vulnerabilities.get(port, []))
            all_vulnerabilities[port] = list(merged_vulns)

            if merged_vulns:
                ports_with_cves.append(port)

        cveportsgraphdata = ports_with_cves

        for port, vulns in all_vulnerabilities.items():
            output += f"\nVulnerabilities for Port {port}:\n"
            if vulns:
                for vuln in vulns:
                    output += f" - {vuln}\n"
                    all_cve_ids.append(vuln)
            else:
                output += " - None found\n"

        output += "\n---------------------------------------------------------------------\n\n"
        gui_queue.put(output)
        search_exploitdb(all_cve_ids)
    except Exception as e:
        gui_queue.put(f"An error occurred during analysis: {e}\n")
    finally:
        progress_bar.stop()

    if all_cve_ids:
        threadedprocess_cve_ids(all_cve_ids)

def display_exploit_details(cve_id):
    api_url = f"https://cve.circl.lu/api/cve/{cve_id}"
    response = requests.get(api_url)
    if response.status_code == 200:
        cve_details = response.json()
    else:
        gui_queue.put(f"\nFailed to retrieve CVE details for {cve_id}. Please check the CVE ID and your internet connection.\n")
        return

    if not cve_details:
        gui_queue.put(f"No details available to display for {cve_id}.\n")
        return

    output = "\n---------------------------------------------------------------------\n"
    output += f"\nDetails for CVE: {cve_details.get('id')}\n\n"
    output += "\n"
    output += f"Summary: {cve_details.get('summary')}\n"
    output += "\n---------------------------------------------------------------------\n"

    if 'references' in cve_details and cve_details['references']:
        output += "\nReferences:\n"
        for ref in cve_details['references'][:5]:
            output += f"\n{ref}\n"
    else:
        output += "\nNo references available.\n"

    output += "\n---------------------------------------------------------------------\n"
    output += f"\nSearching GitHub and Google for potential exploits for {cve_details.get('id')}...\n"
    search_url = f"https://github.com/search?q={cve_details.get('id')}+exploit&type=Repositories"
    output += f"\nCheck out GitHub search results at: {search_url}\n"
    query = f"https://www.google.com/search?q={cve_details.get('id')}+exploit"
    output += f"\nTo search for more exploit details on Google use this query: {query}\n"
    output += "\n---------------------------------------------------------------------\n"
    gui_queue.put(output)

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
        gui_queue.put("Please insert an IP to get info on.\n")
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
        gui_queue.put(resultsipinfo + "\n\n")

def threadedipinfo():
    analysis_thread = Thread(target=ipinfo)
    analysis_thread.start()

def get_ip_website():
    domain = website_entry.get().strip()
    if domain.startswith("http://"):
        domain = domain[7:]
    elif domain.startswith("https://"):
        domain = domain[8:]

    try:
        ip_address = socket.gethostbyname(domain)
        gui_queue.put(ip_address + "\n")
    except socket.gaierror as e:
        gui_queue.put(f"Error getting IP for {domain}: {e}\n")

def threadedwebsiteipinfo():
    analysis_thread = Thread(target=get_ip_website)
    analysis_thread.start()

def findipcam():
    ipcamip = Ip_entry.get()
    output = "If an IP Cam exists it may be at this link\n"
    ipcaminfolink1 = f"http://{ipcamip}:80"
    ipcaminfolink2 = f"http://{ipcamip}:443"
    ipcaminfolink3 = f"http://{ipcamip}:554"
    output += ipcaminfolink1 + "\n"
    output += ipcaminfolink2 + "\n"
    output += ipcaminfolink3 + "\n"
    output += "\n------------------------------------\n"
    gui_queue.put(output)

def clear_view():
    results_text.delete(1.0, tk.END)

def exitprogram():
    sys.exit()

def process_gui_queue():
    try:
        while True:
            result = gui_queue.get_nowait()
            results_text.insert(tk.END, result)
            results_text.see(tk.END)
    except queue.Empty:
        pass
    root.after(100, process_gui_queue)

#GUI
root = tk.Tk()
root.title("SpiderCrawler")
root.geometry("1050x950")
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
timing = tk.StringVar(value="T4")  #default timing is T4

#boxes for different scanning options
Checkbutton(ipframe, text="Fast Scan (Most Common Ports): -F", var=options['fast_scan']).pack(anchor='w')
Checkbutton(ipframe, text="Show Only Open Ports: --open", var=options['show_open']).pack(anchor='w')
Checkbutton(ipframe, text="Version Detection: -sV", var=options['version_detection']).pack(anchor='w')
Checkbutton(ipframe, text="OS Detection: -O", var=options['os_detection']).pack(anchor='w')
Checkbutton(ipframe, text="Script Scan (Default Scripts): -sC", var=options['script_scan']).pack(anchor='w')
Checkbutton(ipframe, text="Aggressive Scan: -A", var=options['aggressive_scan']).pack(anchor='w')
Checkbutton(ipframe, text="No Ping (Skip Discovery): -Pn", var=options['no_ping']).pack(anchor='w')
Checkbutton(ipframe, text="Stealth SYN Scan: -sS", var=options['stealth_scan']).pack(anchor='w')
Checkbutton(ipframe, text="UDP Scan: -sU", var=options['udp_scan']).pack(anchor='w')
Checkbutton(ipframe, text="Vulnerability Scan: --script vulners", var=options['vulnerability_scan']).pack(anchor='w')

#timing options
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

#progress bar
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

process_gui_queue()
root.mainloop()
