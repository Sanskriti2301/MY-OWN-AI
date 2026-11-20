package com.yourownai;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

@SpringBootApplication
@RestController
@CrossOrigin(origins = "*")
public class YourOwnAiApplication {
    private static final int DIMS = 16;
    private final VectorDB db = new VectorDB(DIMS);
    private final DocumentDB docDB = new DocumentDB();
    private final OllamaClient ollama = new OllamaClient();

    public static void main(String[] args) {
        SpringApplication.run(YourOwnAiApplication.class, args);
    }

    public YourOwnAiApplication() { loadDemo(db); }

    // --------------------------- HTTP API ---------------------------

    @GetMapping("/search")
    public Map<String,Object> search(@RequestParam String v,
            @RequestParam(defaultValue="5") int k,
            @RequestParam(defaultValue="cosine") String metric,
            @RequestParam(defaultValue="hnsw") String algo) {
        float[] q = parseVector(v);
        requireDims(q, DIMS);
        VectorDB.SearchOut out = db.search(q, k, metric, algo);
        List<Map<String,Object>> results = new ArrayList<>();
        for (VectorDB.Hit h : out.hits) {
            results.add(Map.of("id",h.id,"metadata",h.meta,"category",h.cat,
                    "distance",h.dist,"embedding",h.emb));
        }
        return Map.of("results",results,"latencyUs",out.us,"algo",out.algo,"metric",out.metric);
    }

    public record InsertRequest(String metadata, String category, List<Float> embedding) {}
    @PostMapping("/insert")
    public Map<String,Object> insert(@RequestBody InsertRequest body) {
        if (body.metadata()==null || body.metadata().isBlank() || body.embedding()==null)
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"invalid body");
        float[] emb = toArray(body.embedding()); requireDims(emb,DIMS);
        return Map.of("id",db.insert(body.metadata(), body.category()==null?"":body.category(), emb, Distances::cosine));
    }

    @DeleteMapping("/delete/{id}")
    public Map<String,Object> delete(@PathVariable int id) { return Map.of("ok",db.remove(id)); }

    @GetMapping("/items")
    public List<Map<String,Object>> items() {
        return db.all().stream().map(v -> Map.<String,Object>of("id",v.id,"metadata",v.metadata,
                "category",v.category,"embedding",v.emb)).toList();
    }

    @GetMapping("/benchmark")
    public Map<String,Object> benchmark(@RequestParam String v,
            @RequestParam(defaultValue="5") int k,
            @RequestParam(defaultValue="cosine") String metric) {
        float[] q=parseVector(v); requireDims(q,DIMS);
        VectorDB.BenchOut b=db.benchmark(q,k,metric);
        return Map.of("bruteforceUs",b.bfUs,"kdtreeUs",b.kdUs,"hnswUs",b.hnswUs,"itemCount",b.n);
    }

    @GetMapping("/hnsw-info")
    public HNSW.GraphInfo hnswInfo() { return db.hnswInfo(); }

    public record DocumentRequest(String title, String text) {}
    @PostMapping("/doc/insert")
    public Map<String,Object> insertDocument(@RequestBody DocumentRequest request) {
        if (blank(request.title()) || blank(request.text()))
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"need title and text");
        List<String> chunks=chunkText(request.text(),250,30);
        List<Integer> ids=new ArrayList<>();
        for(int i=0;i<chunks.size();i++) {
            float[] emb=ollama.embed(chunks.get(i));
            if(emb.length==0) throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                    "Ollama unavailable. Install Ollama, then pull nomic-embed-text and llama3.2.");
            String title=chunks.size()>1?request.title()+" ["+(i+1)+"/"+chunks.size()+"]":request.title();
            ids.add(docDB.insert(title,chunks.get(i),emb));
        }
        return Map.of("ids",ids,"chunks",chunks.size(),"dims",docDB.getDims());
    }

    @DeleteMapping("/doc/delete/{id}")
    public Map<String,Object> deleteDocument(@PathVariable int id) { return Map.of("ok",docDB.remove(id)); }

    @GetMapping("/doc/list")
    public List<Map<String,Object>> documents() {
        return docDB.all().stream().map(d -> Map.<String,Object>of("id",d.id,"title",d.title,
                "preview",d.text.length()>120?d.text.substring(0,120)+"…":d.text,
                "words",d.text.trim().isEmpty()?0:d.text.trim().split("\\s+").length)).toList();
    }

    public record QuestionRequest(String question, Integer k) {}
    @PostMapping("/doc/search")
    public Map<String,Object> documentSearch(@RequestBody QuestionRequest request) {
        if(blank(request.question())) throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"need question");
        float[] q=ollama.embed(request.question());
        if(q.length==0) throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,"Ollama unavailable");
        List<DocumentDB.Hit> hits=docDB.search(q,request.k()==null?3:request.k(),0.7f);
        return Map.of("contexts",hits.stream().map(h->Map.of("id",h.doc.id,"title",h.doc.title,"distance",h.distance)).toList());
    }

    @PostMapping("/doc/ask")
    public Map<String,Object> ask(@RequestBody QuestionRequest request) {
        if(blank(request.question())) throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"need question");
        float[] q=ollama.embed(request.question());
        if(q.length==0) throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,"Ollama unavailable");
        List<DocumentDB.Hit> hits=docDB.search(q,request.k()==null?3:request.k(),0.7f);
        StringBuilder context=new StringBuilder();
        for(int i=0;i<hits.size();i++) context.append("[").append(i+1).append("] ")
                .append(hits.get(i).doc.title).append(":\n").append(hits.get(i).doc.text).append("\n\n");
        String prompt="You are a helpful assistant. Answer the user's question directly. "+
                "Use the provided context if it contains relevant information. "+
                "If it doesn't, just use your own general knowledge. "+
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "+
                "Just answer the question naturally.\n\nContext:\n"+context+
                "Question: "+request.question()+"\n\nAnswer:";
        String answer=ollama.generate(prompt);
        List<Map<String,Object>> contexts=hits.stream().map(h->Map.<String,Object>of(
                "id",h.doc.id,"title",h.doc.title,"text",h.doc.text,"distance",h.distance)).toList();
        return Map.of("answer",answer,"model",ollama.genModel,"contexts",contexts,"docCount",docDB.size());
    }

    @GetMapping("/status")
    public Map<String,Object> status() {
        return Map.of("ollamaAvailable",ollama.isAvailable(),"embedModel",ollama.embedModel,
                "genModel",ollama.genModel,"docCount",docDB.size(),"docDims",docDB.getDims(),
                "demoDims",DIMS,"demoCount",db.size());
    }

    @GetMapping("/stats")
    public Map<String,Object> stats() {
        return Map.of("count",db.size(),"dims",DIMS,
                "algorithms",List.of("bruteforce","kdtree","hnsw"),
                "metrics",List.of("euclidean","cosine","manhattan"));
    }

    // --------------------------- Utilities ---------------------------

    private static boolean blank(String s) { return s==null || s.isBlank(); }
    private static void requireDims(float[] v,int dims) {
        if(v.length!=dims) throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"need "+dims+"D vector");
    }
    private static float[] toArray(List<Float> list) {
        float[] a=new float[list.size()]; for(int i=0;i<a.length;i++) a[i]=list.get(i); return a;
    }
    private static float[] parseVector(String s) {
        if(s==null || s.isBlank()) return new float[0];
        try { String[] p=s.split(","); float[] a=new float[p.length]; for(int i=0;i<p.length;i++) a[i]=Float.parseFloat(p[i].trim()); return a; }
        catch(NumberFormatException e) { throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"invalid vector"); }
    }
    private static List<String> chunkText(String text,int chunkWords,int overlapWords) {
        String trimmed=text.trim(); if(trimmed.isEmpty()) return List.of();
        String[] words=trimmed.split("\\s+");
        if(words.length<=chunkWords) return List.of(text);
        List<String> chunks=new ArrayList<>(); int step=chunkWords-overlapWords;
        if(step<=0) throw new IllegalArgumentException("overlapWords must be smaller than chunkWords");
        for(int i=0;i<words.length;i+=step) {
            chunks.add(String.join(" ",Arrays.copyOfRange(words,i,Math.min(i+chunkWords,words.length))));
            if(i+chunkWords>=words.length) break;
        }
        return chunks;
    }

    // --------------------------- Data model ---------------------------

    static class VectorItem {
        final int id; final String metadata,category; final float[] emb;
        VectorItem(int id,String metadata,String category,float[] emb) {
            this.id=id; this.metadata=metadata; this.category=category; this.emb=emb.clone();
        }
    }
    @FunctionalInterface interface DistFn { float distance(float[] a,float[] b); }

    static class Distances {
        static float euclidean(float[] a,float[] b) {
            checkSame(a,b); float sum=0; for(int i=0;i<a.length;i++){float d=a[i]-b[i];sum+=d*d;} return (float)Math.sqrt(sum);
        }
        static float cosine(float[] a,float[] b) {
            checkSame(a,b); float dot=0,na=0,nb=0;
            for(int i=0;i<a.length;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
            if(na<1e-9f||nb<1e-9f)return 1f;
            return 1f-dot/((float)Math.sqrt(na)*(float)Math.sqrt(nb));
        }
        static float manhattan(float[] a,float[] b) {
            checkSame(a,b); float sum=0;for(int i=0;i<a.length;i++)sum+=Math.abs(a[i]-b[i]);return sum;
        }
        static void checkSame(float[] a,float[] b) {
            if(a.length!=b.length) throw new IllegalArgumentException("Vector dimensions differ");
        }
        static DistFn get(String metric) {
            return switch(metric==null?"euclidean":metric.toLowerCase(Locale.ROOT)) {
                case "cosine" -> Distances::cosine;
                case "manhattan" -> Distances::manhattan;
                default -> Distances::euclidean;
            };
        }
    }

    record Pair(float distance,int id) {}
    static Comparator<Pair> ascendingPair() {
        return Comparator.comparingDouble(Pair::distance).thenComparingInt(Pair::id);
    }

    static class BruteForce {
        final List<VectorItem> items=new ArrayList<>();
        void insert(VectorItem v){items.add(v);}
        void remove(int id){items.removeIf(v->v.id==id);}
        List<Pair> knn(float[] q,int k,DistFn dist) {
            if(k<=0)return List.of();
            return items.stream().map(v->new Pair(dist.distance(q,v.emb),v.id))
                    .sorted(ascendingPair()).limit(k).toList();
        }
    }

    static class KDTree {
        static class Node { VectorItem item; Node left,right; Node(VectorItem v){item=v;} }
        Node root; final int dims;
        KDTree(int dims){this.dims=dims;}
        Node insert(Node n,VectorItem v,int depth) {
            if(n==null)return new Node(v); int axis=depth%dims;
            if(v.emb[axis]<n.item.emb[axis])n.left=insert(n.left,v,depth+1);
            else n.right=insert(n.right,v,depth+1); return n;
        }
        void insert(VectorItem v){root=insert(root,v,0);}
        void search(Node n,float[] q,int k,int depth,DistFn dist,PriorityQueue<Pair> heap) {
            if(n==null||k<=0)return;
            float d=dist.distance(q,n.item.emb);
            if(heap.size()<k||d<heap.peek().distance()) {
                heap.offer(new Pair(d,n.item.id)); if(heap.size()>k)heap.poll();
            }
            int axis=depth%dims; float diff=q[axis]-n.item.emb[axis];
            Node near=diff<0?n.left:n.right, far=diff<0?n.right:n.left;
            search(near,q,k,depth+1,dist,heap);
            if(heap.size()<k||Math.abs(diff)<heap.peek().distance())search(far,q,k,depth+1,dist,heap);
        }
        List<Pair> knn(float[] q,int k,DistFn dist) {
            PriorityQueue<Pair> heap=new PriorityQueue<>(Comparator.comparingDouble(Pair::distance).thenComparingInt(Pair::id).reversed());
            search(root,q,k,0,dist,heap); List<Pair> out=new ArrayList<>(heap);out.sort(ascendingPair());return out;
        }
        void rebuild(Collection<VectorItem> items){root=null;for(VectorItem v:items)insert(v);}
    }

    // HNSW graph, with the same M/M0, seeded random level selection and ef-search strategy as source.
    static class HNSW {
        static class Node { VectorItem item; int maxLayer; List<List<Integer>> neighbors;
            Node(VectorItem item,int level){this.item=item;maxLayer=level;neighbors=new ArrayList<>();for(int i=0;i<=level;i++)neighbors.add(new ArrayList<>());}
        }
        static class GraphInfo {
            public int topLayer,nodeCount;
            public List<Integer> nodesPerLayer=new ArrayList<>(),edgesPerLayer=new ArrayList<>();
            public List<Map<String,Object>> nodes=new ArrayList<>(),edges=new ArrayList<>();
        }
        final Map<Integer,Node> graph=new HashMap<>();
        final int M,M0,efBuild; final float mL;
        int topLayer=-1,entryPoint=-1;
        final Random random=new Random(42);
        HNSW(){this(16,200);}
        HNSW(int m,int ef){M=m;M0=2*m;efBuild=ef;mL=1f/(float)Math.log(m);}
        int randomLevel(){double u=Math.max(random.nextDouble(),1e-12);return (int)Math.floor(-Math.log(u)*mL);}
        List<Pair> searchLayer(float[] q,int ep,int ef,int layer,DistFn dist) {
            if(!graph.containsKey(ep))return List.of();
            Comparator<Pair> asc=ascendingPair();
            PriorityQueue<Pair> candidates=new PriorityQueue<>(asc);
            PriorityQueue<Pair> found=new PriorityQueue<>(asc.reversed());
            Set<Integer> visited=new HashSet<>();
            float d0=dist.distance(q,graph.get(ep).item.emb);
            Pair start=new Pair(d0,ep);visited.add(ep);candidates.offer(start);found.offer(start);
            while(!candidates.isEmpty()) {
                Pair current=candidates.poll();
                if(found.size()>=ef&&current.distance()>found.peek().distance())break;
                Node node=graph.get(current.id());
                if(node==null||layer>=node.neighbors.size())continue;
                for(int nid:node.neighbors.get(layer)) {
                    if(!visited.add(nid)||!graph.containsKey(nid))continue;
                    float nd=dist.distance(q,graph.get(nid).item.emb);
                    if(found.size()<ef||nd<found.peek().distance()) {
                        Pair p=new Pair(nd,nid);candidates.offer(p);found.offer(p);
                        if(found.size()>ef)found.poll();
                    }
                }
            }
            List<Pair> result=new ArrayList<>(found);result.sort(asc);return result;
        }
        List<Integer> selectNeighbors(List<Pair> candidates,int maxM) {
            return candidates.stream().limit(maxM).map(Pair::id).collect(Collectors.toCollection(ArrayList::new));
        }
        void insert(VectorItem item,DistFn dist) {
            int id=item.id,level=randomLevel();graph.put(id,new Node(item,level));
            if(entryPoint==-1){entryPoint=id;topLayer=level;return;}
            int ep=entryPoint;
            for(int lc=topLayer;lc>level;lc--) {
                Node e=graph.get(ep);
                if(e!=null&&lc<e.neighbors.size()){List<Pair>w=searchLayer(item.emb,ep,1,lc,dist);if(!w.isEmpty())ep=w.get(0).id();}
            }
            for(int lc=Math.min(topLayer,level);lc>=0;lc--) {
                List<Pair>w=searchLayer(item.emb,ep,efBuild,lc,dist);
                int maxM=lc==0?M0:M;List<Integer> selected=selectNeighbors(w,maxM);
                graph.get(id).neighbors.set(lc,selected);
                for(int nid:selected) {
                    Node neighbor=graph.get(nid);if(neighbor==null)continue;
                    while(neighbor.neighbors.size()<=lc)neighbor.neighbors.add(new ArrayList<>());
                    List<Integer> conn=neighbor.neighbors.get(lc);conn.add(id);
                    if(conn.size()>maxM) {
                        List<Pair> ds=new ArrayList<>();
                        for(int c:conn)if(graph.containsKey(c))ds.add(new Pair(dist.distance(neighbor.item.emb,graph.get(c).item.emb),c));
                        ds.sort(ascendingPair());conn.clear();
                        for(int i=0;i<Math.min(maxM,ds.size());i++)conn.add(ds.get(i).id());
                    }
                }
                if(!w.isEmpty())ep=w.get(0).id();
            }
            if(level>topLayer){topLayer=level;entryPoint=id;}
        }
        List<Pair> knn(float[] q,int k,int ef,DistFn dist) {
            if(entryPoint==-1||k<=0)return List.of();int ep=entryPoint;
            for(int lc=topLayer;lc>0;lc--) {
                Node e=graph.get(ep);
                if(e!=null&&lc<e.neighbors.size()){List<Pair>w=searchLayer(q,ep,1,lc,dist);if(!w.isEmpty())ep=w.get(0).id();}
            }
            List<Pair>w=searchLayer(q,ep,Math.max(ef,k),0,dist);
            return new ArrayList<>(w.subList(0,Math.min(k,w.size())));
        }
        void remove(int id) {
            if(!graph.containsKey(id))return;
            for(Node n:graph.values())for(List<Integer> layer:n.neighbors)layer.removeIf(x->x==id);
            if(entryPoint==id){entryPoint=-1;for(int other:graph.keySet())if(other!=id){entryPoint=other;break;}
                topLayer=-1;if(entryPoint!=-1)topLayer=graph.get(entryPoint).maxLayer;}
            graph.remove(id);
        }
        GraphInfo getInfo() {
            GraphInfo gi=new GraphInfo();gi.topLayer=topLayer;gi.nodeCount=graph.size();
            int maxL=Math.max(topLayer+1,1);
            for(int i=0;i<maxL;i++){gi.nodesPerLayer.add(0);gi.edgesPerLayer.add(0);}
            for(var entry:graph.entrySet()) {
                int id=entry.getKey();Node n=entry.getValue();
                gi.nodes.add(Map.of("id",id,"metadata",n.item.metadata,"category",n.item.category,"maxLyr",n.maxLayer));
                for(int l=0;l<=n.maxLayer&&l<maxL;l++) {
                    gi.nodesPerLayer.set(l,gi.nodesPerLayer.get(l)+1);
                    if(l<n.neighbors.size())for(int nid:n.neighbors.get(l))if(id<nid) {
                        gi.edgesPerLayer.set(l,gi.edgesPerLayer.get(l)+1);
                        gi.edges.add(Map.of("src",id,"dst",nid,"lyr",l));
                    }
                }
            }
            return gi;
        }
        int size(){return graph.size();}
    }

    static class VectorDB {
        static class Hit {int id;String meta,cat;float[] emb;float dist;
            Hit(int id,String m,String c,float[]e,float d){this.id=id;meta=m;cat=c;emb=e;dist=d;}}
        static class SearchOut {List<Hit> hits=new ArrayList<>();long us;String algo,metric;}
        record BenchOut(long bfUs,long kdUs,long hnswUs,int n){}
        final Map<Integer,VectorItem> store=new LinkedHashMap<>();
        final BruteForce bf=new BruteForce();final KDTree kd;final HNSW hnsw=new HNSW(16,200);
        final int dims;int nextId=1;
        VectorDB(int dims){this.dims=dims;kd=new KDTree(dims);}
        synchronized int insert(String meta,String cat,float[] emb,DistFn dist) {
            requireDims(emb,dims);VectorItem v=new VectorItem(nextId++,meta,cat,emb);store.put(v.id,v);
            bf.insert(v);kd.insert(v);hnsw.insert(v,dist);return v.id;
        }
        synchronized boolean remove(int id) {
            if(store.remove(id)==null)return false;bf.remove(id);hnsw.remove(id);kd.rebuild(store.values());return true;
        }
        synchronized SearchOut search(float[] q,int k,String metric,String algo) {
            DistFn dist=Distances.get(metric);long start=System.nanoTime();List<Pair> raw;
            raw=switch(algo){case "bruteforce"->bf.knn(q,k,dist);case "kdtree"->kd.knn(q,k,dist);default->hnsw.knn(q,k,50,dist);};
            SearchOut out=new SearchOut();out.us=(System.nanoTime()-start)/1000;out.algo=algo;out.metric=metric;
            for(Pair p:raw){VectorItem v=store.get(p.id());if(v!=null)out.hits.add(new Hit(v.id,v.metadata,v.category,v.emb,p.distance()));}
            return out;
        }
        synchronized BenchOut benchmark(float[]q,int k,String metric) {
            DistFn d=Distances.get(metric);long t=System.nanoTime();bf.knn(q,k,d);long b=(System.nanoTime()-t)/1000;
            t=System.nanoTime();kd.knn(q,k,d);long kdUs=(System.nanoTime()-t)/1000;
            t=System.nanoTime();hnsw.knn(q,k,50,d);long h=(System.nanoTime()-t)/1000;
            return new BenchOut(b,kdUs,h,store.size());
        }
        synchronized List<VectorItem> all(){return new ArrayList<>(store.values());}
        synchronized HNSW.GraphInfo hnswInfo(){return hnsw.getInfo();}
        synchronized int size(){return store.size();}
    }

    static class DocItem {final int id;final String title,text;final float[] emb;
        DocItem(int id,String title,String text,float[] emb){this.id=id;this.title=title;this.text=text;this.emb=emb.clone();}}
    static class DocumentDB {
        record Hit(float distance,DocItem doc){}
        final Map<Integer,DocItem> store=new LinkedHashMap<>();final HNSW hnsw=new HNSW(16,200);
        final BruteForce bf=new BruteForce();int nextId=1,dims=0;
        synchronized int insert(String title,String text,float[] emb) {
            if(dims==0)dims=emb.length;else if(dims!=emb.length)throw new IllegalArgumentException("Embedding dimensions changed");
            DocItem d=new DocItem(nextId++,title,text,emb);store.put(d.id,d);
            VectorItem vi=new VectorItem(d.id,title,"doc",emb);hnsw.insert(vi,Distances::cosine);bf.insert(vi);return d.id;
        }
        synchronized List<Hit> search(float[]q,int k,float maxDist) {
            if(store.isEmpty())return List.of();List<Pair> raw=store.size()<10?bf.knn(q,k,Distances::cosine):hnsw.knn(q,k,50,Distances::cosine);
            List<Hit> out=new ArrayList<>();for(Pair p:raw){DocItem d=store.get(p.id());if(d!=null&&p.distance()<=maxDist)out.add(new Hit(p.distance(),d));}return out;
        }
        synchronized boolean remove(int id){if(store.remove(id)==null)return false;hnsw.remove(id);bf.remove(id);return true;}
        synchronized List<DocItem> all(){return new ArrayList<>(store.values());}
        synchronized int size(){return store.size();}
        synchronized int getDims(){return dims;}
    }

    static class OllamaClient {
        final String base=System.getenv().getOrDefault("OLLAMA_URL","http://127.0.0.1:11434");
        final HttpClient client=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
        final ObjectMapper mapper=new ObjectMapper();
        String embedModel=System.getenv().getOrDefault("OLLAMA_EMBED_MODEL","nomic-embed-text");
        String genModel=System.getenv().getOrDefault("OLLAMA_GENERATE_MODEL","llama3.2");
        boolean isAvailable() {
            try {HttpRequest r=HttpRequest.newBuilder(URI.create(base+"/api/tags")).timeout(Duration.ofSeconds(3)).GET().build();
                return client.send(r,HttpResponse.BodyHandlers.ofString()).statusCode()==200;}
            catch(Exception e){return false;}
        }
        float[] embed(String text) {
            try {
                String body=mapper.writeValueAsString(Map.of("model",embedModel,"prompt",text));
                HttpRequest r=HttpRequest.newBuilder(URI.create(base+"/api/embeddings")).timeout(Duration.ofSeconds(35))
                        .header("Content-Type","application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build();
                HttpResponse<String> response=client.send(r,HttpResponse.BodyHandlers.ofString());
                if(response.statusCode()!=200)return new float[0];
                JsonNode arr=mapper.readTree(response.body()).path("embedding");
                if(!arr.isArray())return new float[0];float[] out=new float[arr.size()];
                for(int i=0;i<out.length;i++)out[i]=(float)arr.get(i).asDouble();return out;
            }catch(Exception e){return new float[0];}
        }
        String generate(String prompt) {
            try {
                String body=mapper.writeValueAsString(Map.of("model",genModel,"prompt",prompt,"stream",false));
                HttpRequest r=HttpRequest.newBuilder(URI.create(base+"/api/generate")).timeout(Duration.ofSeconds(190))
                        .header("Content-Type","application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build();
                HttpResponse<String> response=client.send(r,HttpResponse.BodyHandlers.ofString());
                if(response.statusCode()!=200)return "ERROR: Ollama unavailable. Run: ollama serve";
                return mapper.readTree(response.body()).path("response").asText("");
            }catch(Exception e){return "ERROR: Ollama unavailable. Run: ollama serve";}
        }
    }

    // Original 20 demo records, 16 dimensions each.
    private static void loadDemo(VectorDB db) {
        DistFn d=Distances::cosine;
        db.insert("Linked List: nodes connected by pointers","cs",v(.90f,.85f,.72f,.68f,.12f,.08f,.15f,.10f,.05f,.08f,.06f,.09f,.07f,.11f,.08f,.06f),d);
        db.insert("Binary Search Tree: O(log n) search and insert","cs",v(.88f,.82f,.78f,.74f,.15f,.10f,.08f,.12f,.06f,.07f,.08f,.05f,.09f,.06f,.07f,.10f),d);
        db.insert("Dynamic Programming: memoization overlapping subproblems","cs",v(.82f,.76f,.88f,.80f,.20f,.18f,.12f,.09f,.07f,.06f,.08f,.07f,.08f,.09f,.06f,.07f),d);
        db.insert("Graph BFS and DFS: breadth and depth first traversal","cs",v(.85f,.80f,.75f,.82f,.18f,.14f,.10f,.08f,.06f,.09f,.07f,.06f,.10f,.08f,.09f,.07f),d);
        db.insert("Hash Table: O(1) lookup with collision chaining","cs",v(.87f,.78f,.70f,.76f,.13f,.11f,.09f,.14f,.08f,.07f,.06f,.08f,.07f,.10f,.08f,.09f),d);
        db.insert("Calculus: derivatives integrals and limits","math",v(.12f,.15f,.18f,.10f,.91f,.86f,.78f,.72f,.08f,.06f,.07f,.09f,.07f,.08f,.06f,.10f),d);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors","math",v(.20f,.18f,.15f,.12f,.88f,.90f,.82f,.76f,.09f,.07f,.08f,.06f,.10f,.07f,.08f,.09f),d);
        db.insert("Probability: distributions random variables Bayes theorem","math",v(.15f,.12f,.20f,.18f,.84f,.80f,.88f,.82f,.07f,.08f,.06f,.10f,.09f,.06f,.09f,.08f),d);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography","math",v(.22f,.16f,.14f,.20f,.80f,.85f,.76f,.90f,.08f,.09f,.07f,.06f,.08f,.10f,.07f,.06f),d);
        db.insert("Combinatorics: permutations combinations generating functions","math",v(.18f,.20f,.16f,.14f,.86f,.78f,.84f,.80f,.06f,.07f,.09f,.08f,.06f,.09f,.10f,.07f),d);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes","food",v(.08f,.06f,.09f,.07f,.07f,.08f,.06f,.09f,.90f,.86f,.78f,.72f,.08f,.06f,.09f,.07f),d);
        db.insert("Sushi: vinegared rice raw fish and nori rolls","food",v(.06f,.08f,.07f,.09f,.09f,.06f,.08f,.07f,.86f,.90f,.82f,.76f,.07f,.09f,.06f,.08f),d);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs","food",v(.09f,.07f,.06f,.08f,.08f,.09f,.07f,.06f,.82f,.78f,.90f,.84f,.09f,.07f,.08f,.06f),d);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro","food",v(.07f,.09f,.08f,.06f,.06f,.07f,.09f,.08f,.78f,.82f,.86f,.90f,.06f,.08f,.07f,.09f),d);
        db.insert("Croissant: laminated pastry with buttery flaky layers","food",v(.06f,.07f,.10f,.09f,.10f,.06f,.07f,.10f,.85f,.80f,.76f,.82f,.09f,.07f,.10f,.06f),d);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks","sports",v(.09f,.07f,.08f,.10f,.08f,.09f,.07f,.06f,.08f,.07f,.09f,.06f,.91f,.85f,.78f,.72f),d);
        db.insert("Football: tackles touchdowns field goals and strategy","sports",v(.07f,.09f,.06f,.08f,.09f,.07f,.10f,.08f,.07f,.09f,.08f,.07f,.87f,.89f,.82f,.76f),d);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves","sports",v(.08f,.06f,.09f,.07f,.07f,.08f,.06f,.09f,.09f,.06f,.07f,.08f,.83f,.80f,.88f,.82f),d);
        db.insert("Chess: openings endgames tactics strategic board game","sports",v(.25f,.20f,.22f,.18f,.22f,.18f,.20f,.15f,.06f,.08f,.07f,.09f,.80f,.84f,.78f,.90f),d);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition","sports",v(.06f,.08f,.07f,.09f,.08f,.06f,.09f,.07f,.10f,.08f,.06f,.07f,.85f,.82f,.86f,.80f),d);
    }
    private static float[] v(float... values){return values;}
}
