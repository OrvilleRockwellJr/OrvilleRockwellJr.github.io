import requests, json, random, time
random.seed(1)
E="https://rest.ensembl.org"; H={"Content-Type":"application/json","Accept":"application/json"}
def post(path,body):
    for a in range(5):
        r=requests.post(E+path,headers=H,data=json.dumps(body),timeout=120)
        if r.status_code==200: return r.json()
        if r.status_code==429: time.sleep(2+a*2); continue
        print("ERR",r.status_code,r.text[:200]); time.sleep(2)
    return None
def get(path):
    for a in range(5):
        r=requests.get(E+path,headers=H,timeout=120)
        if r.status_code==200: return r.json()
        time.sleep(2)
    return None
hg=requests.get("https://rest.genenames.org/search/status:Approved+AND+locus_group:%22protein-coding%20gene%22?rows=20000",headers={"Accept":"application/json"},timeout=60).json()
syms=[d["symbol"] for d in hg["response"]["docs"]]
test_genes={"ANPEP","PTPRC","NF2","THY1","CD28","PTPRS"}
random.shuffle(syms); syms=[s for s in syms if s not in test_genes][:420]
look=post("/lookup/symbol/homo_sapiens",{"symbols":syms})
ids=[v["canonical_transcript"].split(".")[0] for v in look.values() if v and v.get("canonical_transcript")]
print("transcripts",len(ids))
cds={}
for i in range(0,len(ids),40):
    r=post("/sequence/id",{"ids":ids[i:i+40],"type":"cds"})
    if r:
        for x in r: cds[x["id"]]=x["seq"]
    print(i,len(cds),flush=True)
# PTPRS exons
t=get("/lookup/id/ENST00000262963?expand=1;content-type=application/json")
ex=sorted(t["Exon"],key=lambda e:e["start"],reverse=(t["strand"]==-1))
exons=[]
for n,e in enumerate(ex,1):
    s=get(f"/sequence/id/{e['id']}?expand_5prime=30;expand_3prime=30;type=genomic")
    exons.append({"exon":n,"id":e["id"],"seq":s["seq"],"len":e["end"]-e["start"]+1})
print("PTPRS exons",len(exons),sum(x["len"] for x in exons))
json.dump({"cds":cds,"ptprs_exons":exons,"ptprs_transcript":"ENST00000262963.11","assembly":"GRCh38"},open("/home/claude/build/data/dna.json","w"))
