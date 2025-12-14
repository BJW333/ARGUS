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
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

SHODAN_API_KEY = 'jxLW13mSmgc5PYI3kK1YqUXWvzGZXpbO'
shodan_api = Shodan(SHODAN_API_KEY)

SESSION = requests.Session()
DEFAULT_TIMEOUT = 10  # seconds

open_ports_list = []
cveportsgraphdata = []
graph_canvas = None
graph_figure = None
gui_queue = queue.Queue()

def ensure_nmap_installed():
    try:
        subprocess.run(["nmap", "-V"], capture_output=True, text=True, timeout=5)
        return True
    except Exception:
        gui_queue.put("Error: nmap not found in PATH. Install nmap.\n")
        return False
    
def ensure_searchsploit_installed():
    try:
        result = subprocess.run(["searchsploit", "--help"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        gui_queue.put("Warning: searchsploit not found; ExploitDB lookups skipped.\n")
        return False
    
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
    ax.set_title("Ports with at least one CVE detected")
    
    #handle the canvas
    if graph_canvas is None:
        graph_canvas = FigureCanvasTkAgg(graph_figure, master=results_frame)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    else:
        graph_canvas.draw()

def show_open_ports_pie_chart():
    global cveportsgraphdata
    global graph_canvas
    global graph_figure

    if not cveportsgraphdata:
        gui_queue.put("No CVE-related open ports yet. Run a scan first.\n")
        return

    # toggle behavior
    if graph_canvas is not None:
        try:
            graph_canvas.get_tk_widget().destroy()
        except Exception:
            pass
        graph_canvas = None
        graph_figure = None
        gui_queue.put("Closed ports graph.\n")
    else:
        display_open_ports_pie_chart(cveportsgraphdata, graph_holder)
        gui_queue.put("Displayed ports graph.\n")

def scan_ip_for_open_ports(ip, options, timing='T4'):
    if not ensure_nmap_installed():
        return [], {}, {}   # <- always 3 items
    
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

    cmd.append(f'-{timing}')

    try:
        print(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except Exception as e:
            print(f"Failed running nmap: {e}")
            return [], {}, {}

        if result.returncode != 0:
            print(f"nmap error: {result.stderr.strip()}")
            return [], {}, {}

        return parse_nmap_vuln_output(result.stdout)
    except Exception as e:
        print(f"Error scanning {ip} for open ports: {e}")
        return [], {}, {}
    
def parse_nmap_vuln_output(nmap_output):
    """
    Returns:
      open_ports: list[str] like ["80","443"]
      nmap_vulnerabilities: dict[str, list[str]] -> {"80":["CVE-..."], ...}
      service_info: dict[str, str] -> {"80": "http nginx 1.18.0 (Ubuntu)", ...}
    """
    open_ports = []
    nmap_vulnerabilities = {}
    service_info = {}

    cve_pattern = re.compile(r"\bCVE-\d{4}-\d{4,7}\b")

    # capture 'product/version/extrainfo' if -sV used:
    #
    # Examples:
    # "80/tcp open  http    nginx 1.18.0 (Ubuntu)"
    # "443/tcp open  https   OpenSSL 1.1.1  httpd"
    svc_line = re.compile(
        r"^(\d+)/(tcp|udp)\s+open\s+(\S+)(?:\s+(.*))?$",
        flags=re.IGNORECASE
    )

    current_port = None

    for raw in nmap_output.splitlines():
        line = raw.strip()

        m = svc_line.match(line)
        if m:
            current_port = m.group(1)
            proto = m.group(2)
            svc = m.group(3) or ""
            rest = (m.group(4) or "").strip()
            desc = svc if not rest else f"{svc} {rest}"
            open_ports.append(current_port)
            service_info[current_port] = desc
            continue

        if current_port and "CVE-" in line:
            matches = cve_pattern.findall(line)
            if matches:
                nmap_vulnerabilities.setdefault(current_port, []).extend(matches)

    # de-dup CVEs per port
    for p in list(nmap_vulnerabilities.keys()):
        nmap_vulnerabilities[p] = sorted(set(nmap_vulnerabilities[p]))

    return open_ports, nmap_vulnerabilities, service_info

def get_vulnerabilities_from_shodan(ip, ports):
    if not shodan_api:
        return {}

    try:
        host_info = shodan_api.host(ip)
    except Exception as e:
        msg = f"Shodan lookup failed for {ip}: {e}"
        print(msg)
        gui_queue.put(msg + "\n")
        return {}

    vulns_by_port = {}  #{"80": ["CVE-..."], ...}
    wanted = set(map(int, ports))

    for service in host_info.get('data', []):
        port = service.get('port')
        if isinstance(port, int) and port in wanted:
            v = service.get('vulns') or {}
            if isinstance(v, dict):
                cves = list(v.keys())
            elif isinstance(v, list):
                cves = v
            else:
                cves = []
            if cves:
                vulns_by_port[str(port)] = sorted(set(cves))

    return vulns_by_port

def search_exploitdb(all_cve_ids):
    if not ensure_searchsploit_installed():
        return {}
    
    exploits = {}
    for cve_id in sorted(set(all_cve_ids)):
        try:
            result = subprocess.run(
                ["searchsploit", "--json", cve_id],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    data = json.loads(result.stdout)
                    exploits[cve_id] = data
                    count = len(data.get("RESULTS_EXPLOIT", []))
                    output = f"Results for {cve_id}: {count} exploits found\n"
                except json.JSONDecodeError:
                    exploits[cve_id] = result.stdout
                    output = f"Results for {cve_id}: (raw JSON parse failed)\n"
            else:
                output = f"No results for {cve_id}.\n"
                exploits[cve_id] = output
        except subprocess.TimeoutExpired:
            output = f"searchsploit timed out for {cve_id}\n"
            exploits[cve_id] = output
        except Exception as e:
            output = f"Error searching ExploitDB for {cve_id}: {e}\n"
            exploits[cve_id] = output
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
        #progress_bar.start(2)
        with open(ip_file, 'r') as file:
            for ip in file:
                ip = ip.strip()
                IP = ip

                progress_bar.start(2) #old line moved to outside the loop

                if IP:
                    output = f"Analyzing IP: {IP}\n---------------------------------\n"

                    #open_ports_list, nmap_vulnerabilities = scan_ip_for_open_ports(IP, options, timing.get())
                    open_ports_list, nmap_vulnerabilities, service_info = scan_ip_for_open_ports(IP, options, timing.get())

                    output += "Open ports:\n"
                    for port in open_ports_list:
                        desc = service_info.get(port, "")
                        output += f" - Port {port} — {desc}\n" if desc else f" - Port {port}\n"
                                    

                    output += "\n---------------------------------------------------------------------\n"

                    #shodan_vulnerabilities = get_vulnerabilities_from_shodan(IP, open_ports_list) #old line
                    shodan_vulnerabilities = get_vulnerabilities_from_shodan(IP, open_ports_list) if shodan_api else {}
                    
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
                    #search_exploitdb(all_cve_ids) #old line    
            if all_cve_ids:
                print("this is all the cve ids analyze file gui", all_cve_ids) #debugging line
                search_exploitdb(sorted(set(all_cve_ids)))
                        
    except FileNotFoundError:
        gui_queue.put(f"File {ip_file} not found.\n")
    except Exception as e:
        gui_queue.put(f"An error occurred: {e}\n")
    finally:
        progress_bar.stop()

    if all_cve_ids:
        gui_queue.put(f"Fetching details for {len(all_cve_ids)} unique CVEs...\n")
        threadedprocess_cve_ids(sorted(set(all_cve_ids)))
    else:
        gui_queue.put("No CVEs to fetch details for.\n")

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
        open_ports_list, nmap_vulnerabilities, service_info = scan_ip_for_open_ports(iptarget, options, timing.get())

        output += "Open ports:\n"
        for port in open_ports_list:
            desc = service_info.get(port, "")
            if desc:
                output += f" - Port {port} — {desc}\n"
            else:
                output += f" - Port {port}\n"

        output += "\n---------------------------------------------------------------------\n"

        shodan_vulnerabilities = get_vulnerabilities_from_shodan(iptarget, open_ports_list) if shodan_api else {}        
        
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
        search_exploitdb(sorted(set(all_cve_ids)))
        print("this is all the cve ids analyze ip gui", all_cve_ids) #debugging line

    except Exception as e:
        gui_queue.put(f"An error occurred during analysis: {e}\n")
    finally:
        progress_bar.stop()

    if all_cve_ids:
        gui_queue.put(f"Fetching details for {len(all_cve_ids)} unique CVEs...\n")
        threadedprocess_cve_ids(sorted(set(all_cve_ids)))
    else:
        gui_queue.put("No CVEs to fetch details for.\n")


def display_exploit_details(cve_id):
    """
    Show CVE details using NVD v2 first, then CIRCL as fallback.
    Adds UA header, optional NVD API key, and robust parsing.
    """
    # ---- endpoints
    nvd_url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    circl_url = f"https://cve.circl.lu/api/cve/{cve_id}"

    def safe_get(url, timeout=DEFAULT_TIMEOUT, tries=3, backoff=0.9):
        last_exc = None
        for attempt in range(1, tries + 1):
            try:
                r = SESSION.get(url, timeout=timeout)
                return r
            except Exception as e:
                last_exc = e
                if attempt < tries:
                    time.sleep(backoff * attempt)
        return None

    # ---------- helper: build GUI text ----------
    def emit_card(cid, published, modified, summary, cwe, cvss3, severity, refs):
        out = []
        out.append("\n---------------------------------------------------------------------\n")
        out.append(f"Details for CVE: {cid}\n")
        out.append(f"Published: {published or 'Unknown'} | Modified: {modified or 'Unknown'}\n")
        if cvss3 is not None:
            sev = f" | Severity: {severity}" if severity else ""
            out.append(f"CVSS v3 Base Score: {cvss3}{sev}\n")
        out.append("\nSummary:\n")
        out.append((summary or "No summary provided.") + "\n")
        out.append("\nWeakness (CWE): " + (cwe or "N/A") + "\n")
        out.append("\nReferences (up to 5):\n")
        if refs:
            for r in refs[:5]:
                out.append(f"- {r}\n")
        else:
            out.append("- No references available.\n")
        out.append("\n---------------------------------------------------------------------\n")
        out.append("Search for public exploits:\n")
        out.append(f"- GitHub: https://github.com/search?q={cid}+exploit&type=Repositories\n")
        out.append(f"- Google: https://www.google.com/search?q={cid}+exploit\n")
        out.append("---------------------------------------------------------------------\n")
        gui_queue.put("".join(out))

    # ---------- NVD v2 first ----------
    r = safe_get(nvd_url, tries=3)
    used_nvd = False
    if r and r.status_code == 200:
        try:
            j = r.json()
            vulns = j.get("vulnerabilities") or []
            if vulns:
                v = vulns[0].get("cve", {})  # structure: [{ "cve": {...}}]
                cid = v.get("id", cve_id)

                # description
                summary = None
                for d in v.get("descriptions", []):
                    if d.get("lang") == "en":
                        summary = d.get("value")
                        break

                # references
                refs = [ref.get("url") for ref in v.get("references", []) if ref.get("url")]

                # dates
                published = v.get("published") or v.get("publishedDate") or "Unknown"
                modified = v.get("lastModified") or "Unknown"

                # CWE (problemTypes may have multiple, pick first English description if present)
                cwe = None
                for pt in v.get("weaknesses", []):
                    for d in pt.get("description", []):
                        if d.get("lang") == "en":
                            cwe = d.get("value")
                            break
                    if cwe:
                        break

                # CVSS v3 (prefer 3.1 then 3.0)
                cvss3 = None
                severity = None
                metrics = v.get("metrics", {})
                for key in ("cvssMetricV31", "cvssMetricV30"):
                    arr = metrics.get(key, [])
                    if arr:
                        data = arr[0].get("cvssData", {})
                        cvss3 = data.get("baseScore")
                        severity = data.get("baseSeverity")
                        break

                emit_card(cid, published, modified, summary, cwe, cvss3, severity, refs)
                used_nvd = True
        except ValueError:
            pass  # fall through to CIRCL

    if used_nvd:
        return
    else:
        if r is None:
            gui_queue.put(f"Caution: NVD v2 request failed for {cve_id} (no response). Falling back to CIRCL…\n")
        else:
            gui_queue.put(f"Caution: NVD v2 returned HTTP {r.status_code} for {cve_id}. Falling back to CIRCL…\n")

    # ---------- CIRCL fallback ----------
    r2 = safe_get(circl_url, tries=2)
    circl = {}
    if r2 and r2.status_code == 200:
        try:
            circl = r2.json() or {}
        except ValueError:
            circl = {}
    else:
        status = r2.status_code if r2 else "no response"
        gui_queue.put(f"CIRCL request issue for {cve_id}: {status}\n")

    if circl:
        cid = circl.get("id") or circl.get("CVE") or cve_id
        summary = circl.get("summary") or circl.get("Summary")
        published = circl.get("Published") or circl.get("published") or "Unknown"
        modified  = circl.get("Modified")  or circl.get("modified")  or "Unknown"
        cwe = circl.get("cwe") or "N/A"
        refs = circl.get("references") or []

        # CVSS v3 variations sometimes present
        metrics = circl.get("cvss3") or circl.get("cvss_3") or circl.get("cvssV3") or {}
        cvss3 = None
        severity = None
        if isinstance(metrics, dict):
            cvss3 = metrics.get("baseScore") or metrics.get("cvssV3", {}).get("baseScore")
            severity = metrics.get("baseSeverity")

        # If CIRCL has no summary, synthesize a minimal one from CPEs or refs
        if not summary:
            cpes = circl.get("vulnerable_configuration") or circl.get("vulnerable_product") or []
            hint = f"Affects {len(cpes)} CPEs" if cpes else "Summary not provided by CIRCL"
            if refs:
                hint += "; see references."
            summary = hint

        emit_card(cid, published, modified, summary, cwe, cvss3, severity, refs)
        return

    # ---------- Nothing useful ----------
    gui_queue.put(
        f"No detailed metadata found for {cve_id} on NVD v2 or CIRCL.\n"
        "Possible causes: very new CVE, API/network issues, or malformed ID.\n"
    )
    # for debugging, try one more CIRCL raw fetch and show tiny snippet
    r_debug = safe_get(circl_url, tries=1)
    if r_debug:
        gui_queue.put(f"CIRCL HTTP {r_debug.status_code}, body length {len(r_debug.text)} bytes.\n")
        if len(r_debug.text) < 2000:
            gui_queue.put(f"Body snippet:\n{r_debug.text[:1200]}\n")
            
            
def fetch_and_display_cve_details(cve_id):
    display_exploit_details(cve_id)

def threadedprocess_cve_ids(cve_ids):
    unique_ids = sorted(set(cve_ids))
    if not unique_ids:
        return
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(fetch_and_display_cve_details, cid) for cid in unique_ids]
        for _ in as_completed(futures):
            pass

def analyze_threadedIPtarget():
    analysis_thread = Thread(target=analyze_ip_gui)
    analysis_thread.start()
    findipcam()

def ipinfo():
    ipinfotarget = Ip_entry.get()
    api = "http://ip-api.com/json/"
    
    if not ipinfotarget:
        gui_queue.put("Please insert an IP to get info on.\n")
        return
    
    try:
        data = SESSION.get(api + ipinfotarget, timeout=DEFAULT_TIMEOUT).json()
    except Exception as e:
        gui_queue.put(f"IP info lookup failed: {e}\n")
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
    if not domain:
        gui_queue.put("Please enter a domain.\n")
        return
    try:
        infos = socket.getaddrinfo(domain, None)
        addrs = sorted({i[4][0] for i in infos})
        gui_queue.put(("\n".join(addrs)) + "\n")
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
    try:
        # cancel pending after callback if present
        try:
            if hasattr(root, "_after_id"):
                root.after_cancel(root._after_id)
        except Exception:
            pass
        root.destroy()
    except Exception:
        sys.exit(0)
        
def process_gui_queue():
    try:
        while True:
            result = gui_queue.get_nowait()
            results_text.insert(tk.END, result)
            results_text.see(tk.END)
    except queue.Empty:
        pass

    # schedule next call only if root still exists
    try:
        if root.winfo_exists():
            # store id so it can be cancelled later
            root._after_id = root.after(100, process_gui_queue)
    except tk.TclError:
        # GUI is gone or Tcl interpreter closed — ignore safely
        pass


#GUI
root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", exitprogram)
root.title("SpiderCrawler")
root.geometry("1100x800")
root.minsize(900, 700)
# Use ttk everywhere for consistent look
style = ttk.Style()
# macOS often defaults to 'aqua'; 'clam' is cleaner and works with dark bg too
try:
    style.theme_use("clam")
except Exception:
    pass

# Base spacing
PADX, PADY = 8, 6
style.configure("TLabel", padding=(2, 2))
style.configure("TCheckbutton", padding=(2, 2))
style.configure("TRadiobutton", padding=(2, 2))
style.configure("TButton", padding=(10, 6))

# Root grid: two columns – left controls (fixed width), right results (expands)
root.columnconfigure(0, weight=0)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)

# ---- LEFT: controls column ----------------------------------------------------
left = ttk.Frame(root, padding=12)
left.grid(row=0, column=0, sticky="n")

# Target + timing section
target_box = ttk.Labelframe(left, text="Target & Scan Options", padding=10)
target_box.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, PADY))
target_box.columnconfigure(1, weight=1)

ttk.Label(target_box, text="IP Target:").grid(row=0, column=0, sticky="w")
Ip_entry = ttk.Entry(target_box, width=18)
Ip_entry.grid(row=0, column=1, sticky="ew", pady=(0, PADY))

timing = tk.StringVar(value="T4")

#options dictionary here
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

# Scan option checkboxes (two columns for neatness)
checks = [
    ("Fast Scan (-F)", 'fast_scan'),
    ("Show Only Open (--open)", 'show_open'),
    ("Version Detection (-sV)", 'version_detection'),
    ("OS Detection (-O)", 'os_detection'),
    ("Default Scripts (-sC)", 'script_scan'),
    ("Aggressive (-A)", 'aggressive_scan'),
    ("No Ping (-Pn)", 'no_ping'),
    ("Stealth SYN (-sS)", 'stealth_scan'),
    ("UDP Scan (-sU)", 'udp_scan'),
    ("Vulners Script", 'vulnerability_scan'),
]
chk_frame = ttk.Frame(target_box)
chk_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, PADY))
for i in range(2):
    chk_frame.columnconfigure(i, weight=1)

r = 0
for i, (label, key) in enumerate(checks):
    c = i % 2
    if c == 0 and i > 0:
        r += 1
    ttk.Checkbutton(chk_frame, text=label, variable=options[key]).grid(row=r, column=c, sticky="w", padx=(0, 12), pady=2)

# Timing radios
radio_frame = ttk.Frame(target_box)
radio_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(PADY, 0))
ttk.Label(radio_frame, text="Timing Template:").grid(row=0, column=0, sticky="w", padx=(0, 12))
for idx, tval in enumerate(["T2", "T3", "T4", "T5"]):
    ttk.Radiobutton(radio_frame, text=tval, value=tval, variable=timing).grid(row=0, column=idx+1, sticky="w", padx=(0, 6))

# Actions row
actions = ttk.Frame(left)
actions.grid(row=1, column=0, sticky="ew", pady=(0, PADY))
actions.columnconfigure(0, weight=1)
ttk.Button(actions, text="IPAnalyze", command=analyze_threadedIPtarget).grid(row=0, column=0, sticky="ew")

aux = ttk.Frame(left)
aux.grid(row=2, column=0, sticky="ew", pady=(0, PADY))
ttk.Button(aux, text="IPInfo", command=threadedipinfo).grid(row=0, column=0, padx=(0, 6))
ttk.Button(aux, text="Show Ports Graph", command=show_open_ports_pie_chart).grid(row=0, column=1)

# Domain tools
domain_box = ttk.Labelframe(left, text="Domain Tools", padding=10)
domain_box.grid(row=3, column=0, sticky="ew", pady=(0, PADY))
domain_box.columnconfigure(1, weight=1)

ttk.Label(domain_box, text="Domain:").grid(row=0, column=0, sticky="w")
website_entry = ttk.Entry(domain_box)
website_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
ttk.Button(domain_box, text="Find Domain IP", command=threadedwebsiteipinfo).grid(row=0, column=2, sticky="e")

# File tools
file_box = ttk.Labelframe(left, text="Batch from File", padding=10)
file_box.grid(row=4, column=0, sticky="ew", pady=(0, PADY))
file_box.columnconfigure(1, weight=1)

ttk.Label(file_box, text="File Path:").grid(row=0, column=0, sticky="w")
file_entry = ttk.Entry(file_box)
file_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
ttk.Button(file_box, text="Browse", command=select_file).grid(row=0, column=2, sticky="e", padx=(0, 6))
ttk.Button(file_box, text="IPFileAnalyze", command=analyze_threaded).grid(row=0, column=3, sticky="e")

# ---- RIGHT: results column ----------------------------------------------------
right = ttk.Frame(root, padding=(0, 12, 12, 12))
right.grid(row=0, column=1, sticky="nsew")
right.columnconfigure(0, weight=1)
right.rowconfigure(2, weight=1)   # the log area expands

# progress bar (row 0)
progress_bar = ttk.Progressbar(right, orient="horizontal", mode="indeterminate", length=280)
progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, PADY))

# graph holder (row 1)
graph_holder = ttk.Frame(right)
graph_holder.grid(row=1, column=0, sticky="nsew", pady=(0, PADY))

# log area (row 2)
resultsframe = ttk.Frame(right)
resultsframe.grid(row=2, column=0, sticky="nsew")
resultsframe.columnconfigure(0, weight=1)
resultsframe.rowconfigure(0, weight=1)

results_text = scrolledtext.ScrolledText(resultsframe, wrap="word")
results_text.grid(row=0, column=0, sticky="nsew")

# bottom bar (row 3)
bottom = ttk.Frame(right)
bottom.grid(row=3, column=0, sticky="ew", pady=(PADY, 0))
ttk.Button(bottom, text="Clear View", command=clear_view).pack(side="left")
ttk.Button(bottom, text="Exit / Cancel", command=exitprogram).pack(side="right")

# Start processing the GUI queue
process_gui_queue()
root.mainloop()
