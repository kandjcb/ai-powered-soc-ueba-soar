import os
"""
UEBA Feature Extraction - FIXED composite aggregation version
================================================================
Your previous version of this script looped through 8 different user
identity fields and ran the FULL aggregation separately for each one,
then merged results - this counts many real events multiple times
(once per matching field), which is a duplication bug, structurally
similar to the overlapping-window bug found earlier.

THE FIX: query ONCE per event category, using a single coalesced
"user identity" that tries each field in priority order and uses
the FIRST one that has a value - never sums across fields.

WHAT'S GENUINELY NEW (not duplication): this version explicitly
queries Windows logon events (event.code 4624 successful, 4625
failed) as their own category - the original 4-function extractor
never queried these directly. This is real, previously-missing
signal, not inflated row count.

Still uses SESSION-BASED grouping (gap detection), not fixed 1h
windows - the 1h window in the broken version was a second,
separate problem, also fixed here.
"""
from elasticsearch import Elasticsearch
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ES_HOST = os.getenv("ES_HOST", "https://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASS = os.environ["ES_PASS"]  # required, set as environment variable
VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "false").lower() == "true"

FILEBEAT_INDEX = "filebeat-*"
WINLOGBEAT_INDEX = "winlogbeat-*"

GAP_THRESHOLD_HOURS = 2
MAX_SESSION_MINUTES = 45
START_TIME = "2026-07-01T00:00:00"
END_TIME = "now"
WORK_HOUR_START = 9
WORK_HOUR_END = 18

es = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS), verify_certs=VERIFY_CERTS, request_timeout=60)


def paginated_search(index, query_body, page_size=1000):
    query_body = dict(query_body)
    query_body["size"] = page_size
    query_body.setdefault("sort", [{"@timestamp": "asc"}])
    all_hits = []
    search_after = None
    while True:
        if search_after:
            query_body["search_after"] = search_after
        res = es.search(index=index, body=query_body)
        hits = res["hits"]["hits"]
        if not hits:
            break
        all_hits.extend(hits)
        search_after = hits[-1]["sort"]
    return all_hits


def coalesce_user(src):
    """Try each identity field in priority order, return the FIRST
    one that has a value. Never combine/sum across fields - this is
    what prevents the duplication bug."""
    candidates = [
        src.get("auditd", {}).get("log", {}).get("AUID"),
        src.get("winlog", {}).get("event_data", {}).get("TargetUserName"),
        src.get("winlog", {}).get("event_data", {}).get("User"),
        src.get("user", {}).get("name"),
    ]
    for c in candidates:
        if c and str(c).strip() and str(c).lower() not in ["none", "null", "-", "unknown"]:
            return str(c).strip()
    return "unknown"


# ============================================================
# FETCH FUNCTIONS - each queried EXACTLY ONCE per category
# ============================================================

def fetch_failed_logins():
    query_body = {"query": {"bool": {"must": [
        {"match_phrase": {"message": "Failed password"}},
        {"range": {"@timestamp": {"gte": START_TIME, "lte": END_TIME}}}
    ]}}}
    hits = paginated_search(FILEBEAT_INDEX, query_body)
    records = []
    for hit in hits:
        src = hit["_source"]
        msg = src.get("message", "")
        user = "unknown"
        if "for " in msg and " from " in msg:
            try:
                user = msg.split("for ")[1].split(" from")[0].strip().replace("invalid user ", "")
            except IndexError:
                pass
        records.append({"timestamp": src["@timestamp"], "user": user,
                         "event_type": "failed_login", "process": None,
                         "tty": None, "ppid": None, "cmdline": ""})
    print(f"  fetched {len(records)} failed_login events (Linux)")
    return records


def fetch_windows_logon_events():
    """NEW: Windows Security log 4624 (success) / 4625 (failure) -
    genuinely not captured by the original extraction."""
    query_body = {"query": {"bool": {"must": [
        {"terms": {"winlog.event_id": [4624, 4625]}},
        {"range": {"@timestamp": {"gte": START_TIME, "lte": END_TIME}}}
    ]}}}
    hits = paginated_search(WINLOGBEAT_INDEX, query_body)
    records = []
    for hit in hits:
        src = hit["_source"]
        user = coalesce_user(src)
        event_id = src.get("winlog", {}).get("event_id")
        event_type = "failed_login" if str(event_id) == "4625" else "successful_login"
        records.append({"timestamp": src["@timestamp"], "user": user,
                         "event_type": event_type, "process": None,
                         "tty": None, "ppid": None, "cmdline": ""})
    print(f"  fetched {len(records)} Windows logon events (4624/4625)")
    return records


def fetch_sysmon_network_connections():
    query_body = {"query": {"bool": {"must": [
        {"match": {"event.code": "3"}},
        {"range": {"@timestamp": {"gte": START_TIME, "lte": END_TIME}}}
    ]}}}
    hits = paginated_search(WINLOGBEAT_INDEX, query_body)
    records = []
    for hit in hits:
        src = hit["_source"]
        user = coalesce_user(src)
        records.append({"timestamp": src["@timestamp"], "user": user,
                         "event_type": "network_connection", "process": None,
                         "tty": None, "ppid": None, "cmdline": ""})
    print(f"  fetched {len(records)} network_connection events")
    return records


def fetch_sysmon_process_creation():
    query_body = {"query": {"bool": {"must": [
        {"match": {"event.code": "1"}},
        {"range": {"@timestamp": {"gte": START_TIME, "lte": END_TIME}}}
    ]}}}
    hits = paginated_search(WINLOGBEAT_INDEX, query_body)
    records = []
    for hit in hits:
        src = hit["_source"]
        user = coalesce_user(src)
        image = src.get("winlog", {}).get("event_data", {}).get("Image", "unknown")
        cmdline = src.get("winlog", {}).get("event_data", {}).get("CommandLine", "")
        records.append({"timestamp": src["@timestamp"], "user": user,
                         "event_type": "process_creation", "process": image,
                         "tty": None, "ppid": None, "cmdline": cmdline})
    print(f"  fetched {len(records)} process_creation events")
    return records


def fetch_auditd_execve():
    query_body = {"query": {"bool": {
        "must": [
            {"match": {"event.module": "auditd"}},
            {"match": {"auditd.log.SYSCALL": "execve"}},
            {"range": {"@timestamp": {"gte": START_TIME, "lte": END_TIME}}}
        ],
        "must_not": [{"match": {"process.name": "systemd-timesyncd"}}]
    }}}
    hits = paginated_search(FILEBEAT_INDEX, query_body)
    records = []
    for hit in hits:
        src = hit["_source"]
        comm = src.get("process", {}).get("name", "unknown")
        tty = src.get("auditd", {}).get("log", {}).get("tty", "unknown")
        ppid = src.get("process", {}).get("ppid", None)
        user = coalesce_user(src)
        records.append({"timestamp": src["@timestamp"], "user": user,
                         "event_type": "command_execution", "process": comm,
                         "tty": tty, "ppid": ppid, "cmdline": ""})
    print(f"  fetched {len(records)} auditd command_execution events")
    return records


# ============================================================
# SESSION DETECTION + FEATURE BUILDING (same as validated version)
# ============================================================

SUSPICIOUS_COMMANDS = ["nc", "netcat", "ncat", "nmap", "hydra", "john", "hashcat",
                        "mimikatz", "rubeus", "wget", "curl", "base64"]
PRIVESC_RECON_PATTERNS = ["localgroup administrators", "query hklm\\sam",
                           "query hklm\\security", "whoami /priv", "whoami /groups"]


def assign_sessions(df):
    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)
    session_ids = []
    current_id = 0
    last_user = None
    last_time = None
    for _, row in df.iterrows():
        if row["user"] != last_user:
            current_id += 1
        elif last_time is not None:
            gap_hours = (row["timestamp"] - last_time).total_seconds() / 3600
            if gap_hours >= GAP_THRESHOLD_HOURS:
                current_id += 1
        session_ids.append(current_id)
        last_user = row["user"]
        last_time = row["timestamp"]
    df["session_id"] = session_ids
    return df

def add_novelty_flags(df):
    df = df.sort_values("timestamp").copy()
    seen = {}
    novel_flags = []
    for _, row in df.iterrows():
        proc = row["process"]
        if proc is None:
            novel_flags.append(False)
            continue
        key = (row["user"], proc)
        if key not in seen:
            seen[key] = True
            novel_flags.append(True)
        else:
            novel_flags.append(False)
    df["is_novel"] = novel_flags
    return df


def build_feature_table(all_records):
    df = pd.DataFrame(all_records)
    if df.empty:
        print("No records found.")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_off_hours"] = ((df["hour_of_day"] < WORK_HOUR_START) | (df["hour_of_day"] > WORK_HOUR_END)).astype(int)
    df["is_suspicious_cmd"] = df["process"].isin(SUSPICIOUS_COMMANDS).fillna(False)
    df["is_orphan"] = (df["tty"] == "(none)") & (df["ppid"] == 1)
    df["cmdline_lower"] = df["cmdline"].astype(str).str.lower()
    df["is_privesc_recon"] = df["cmdline_lower"].apply(lambda c: any(p in c for p in PRIVESC_RECON_PATTERNS))
    df = add_novelty_flags(df)
    df = assign_sessions(df)

    rows = []
    for (user, session_id), group in df.groupby(["user", "session_id"]):
        row = {"user": user, "session_id": session_id,
               "time_window": group["timestamp"].min(),
               "session_end": group["timestamp"].max()}
        counts = group["event_type"].value_counts()
        for event_type in ["failed_login", "successful_login", "network_connection",
                            "process_creation", "command_execution"]:
            row[event_type] = int(counts.get(event_type, 0))
        row["off_hours_ratio"] = float(group["is_off_hours"].mean())
        row["novel_event_count"] = int(group["is_novel"].sum())
        row["orphan_process_count"] = int(group["is_orphan"].sum())
        row["priv_esc_attempt_count"] = int(group["is_privesc_recon"].sum())
        row["suspicious_cmd_count"] = int(group["is_suspicious_cmd"].sum())
        row["unique_processes_count"] = int(group["process"].dropna().nunique())
        row["day_of_week"] = int(group["timestamp"].min().dayofweek)
        row["session_duration_min"] = round((row["session_end"] - row["time_window"]).total_seconds() / 60, 1)
        rows.append(row)

    result = pd.DataFrame(rows)
    print(f"\nSession detection: {len(result)} distinct sessions found")
    return result


if __name__ == "__main__":
    print("Fetching failed logins (Linux)...")
    failed_logins = fetch_failed_logins()
    print("Fetching Windows logon events (4624/4625)...")
    win_logons = fetch_windows_logon_events()
    print("Fetching Sysmon network connections...")
    network_conns = fetch_sysmon_network_connections()
    print("Fetching Sysmon process creation events...")
    process_creation = fetch_sysmon_process_creation()
    print("Fetching auditd events...")
    auditd_events = fetch_auditd_execve()

    all_records = failed_logins + win_logons + network_conns + process_creation + auditd_events
    print(f"\nTotal raw events collected: {len(all_records)}")

    print("Building session-based feature table...")
    features = build_feature_table(all_records)
    print(f"\nFinal feature table: {len(features)} rows")
    print(features[["user", "time_window", "failed_login", "successful_login",
                     "process_creation", "command_execution"]].to_string())

    features.to_csv("ueba_features.csv", index=False)
    print("\nSaved to ueba_features.csv")
