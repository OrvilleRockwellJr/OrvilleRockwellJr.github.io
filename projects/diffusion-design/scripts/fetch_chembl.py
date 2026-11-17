import requests, json, time, sys
B="https://www.ebi.ac.uk/chembl/api/data/activity.json"
T={"CHEMBL2396508":"PTPRS","CHEMBL2146342":"PTPRS_brain","CHEMBL335":"PTPN1","CHEMBL3864":"PTPN11","CHEMBL3521":"PTPRF",
   "CHEMBL3243":"PTPRC","CHEMBL3807":"PTPN2","CHEMBL3166":"PTPN6","CHEMBL2706":"PTPRB","CHEMBL4905":"PTPRG","CHEMBL2889":"PTPN22"}
rows=[]
for tid,name in T.items():
    off=0; n=0
    while True:
        for attempt in range(4):
            try:
                r=requests.get(B,params={"target_chembl_id":tid,"standard_type__in":"IC50,Ki,Kd","standard_units":"nM","limit":1000,"offset":off},timeout=60).json(); break
            except Exception as e: time.sleep(3)
        acts=r["activities"]
        for a in acts:
            if a.get("standard_relation")!="=" or not a.get("canonical_smiles") or a.get("standard_value") is None: continue
            rows.append(dict(target=name,tid=tid,smiles=a["canonical_smiles"],mol=a["molecule_chembl_id"],type=a["standard_type"],nM=float(a["standard_value"])))
            n+=1
        if not r["page_meta"]["next"]: break
        off+=1000
    print(name,tid,n,flush=True)
json.dump(rows,open("/home/claude/build/data/chembl_ptp.json","w"))
print(len(rows))
