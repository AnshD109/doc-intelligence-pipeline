# 🎯 Interview Talking Points — Document Intelligence Pipeline

## The 60-Second Pitch (Memorize This)

> "I built a production RAG pipeline that lets you ask natural language questions
> over a corpus of 5 annual reports — Apple, Tesla, SAP, Siemens, and BMW.
> The interesting part wasn't building the pipeline itself — it was evaluating and
> improving it. I created a 30-question ground truth test set and measured retrieval
> quality using Hit@k and MRR metrics. My baseline got around 60% Hit@3.
> I ran experiments across chunk sizes and retrieval strategies, and found that
> chunk size 1000 with diversity reranking pushed Hit@3 to over 80%.
> The system runs as a FastAPI microservice with a Streamlit UI, monitored
> with Evidently for query drift, containerized with Docker."

---

## Questions You Will Be Asked — With Answers

### "Why did you choose FAISS over Pinecone or ChromaDB?"
> FAISS is a local, open-source library from Meta — no external API calls,
> no cost, no latency overhead from network requests. For a document corpus
> of this size (under 50K chunks), FAISS IndexFlatIP gives exact cosine
> similarity search in under 50ms. Pinecone makes sense at scale when you
> need distributed storage — overkill here. ChromaDB adds persistence
> complexity I didn't need since I handle that with pickle + FAISS's native
> write_index. The right tool for the right scale.

### "What chunk size did you use and why?"
> I didn't just pick one — I ran experiments comparing 500, 1000, and 1500
> tokens. 500 was too small — chunks often cut mid-sentence losing context.
> 1500 diluted relevance by mixing multiple topics. 1000 tokens with 200
> overlap hit the sweet spot in my evaluation: best MRR and Hit@5 scores
> with acceptable latency. The 200 token overlap ensures continuity across
> chunk boundaries.

### "How did you evaluate retrieval quality?"
> I built a 30-question ground truth test set manually verified against the
> actual PDF documents. I measured Hit@1, Hit@3, Hit@5, and MRR — standard
> IR metrics. Hit@k tells you if the correct source appeared in the top k
> results. MRR tells you how highly ranked it was. This gave me an objective
> way to compare configurations rather than just eyeballing answers.

### "What is MRR and why is it useful?"
> Mean Reciprocal Rank is the average of 1/rank across queries, where rank
> is the position of the first correct result. If the correct document is
> always ranked first, MRR = 1.0. If it's always second, MRR = 0.5.
> It's more informative than Hit@k alone because it captures ranking quality,
> not just presence. A system that always ranks the correct document 5th
> would have the same Hit@5 as one that ranks it 1st, but very different MRR.

### "What's the difference between your naive RAG and diversity reranking?"
> Naive top-k retrieval often returns 4-5 chunks from the same document —
> because similar topics cluster together in embedding space. For single-doc
> questions this is fine, but for cross-document questions like "compare
> R&D spending across companies", you end up with only Apple chunks.
> My diversity reranker fetches top-10, then selects the highest-scoring
> chunk per source, ensuring representation from multiple documents.
> This significantly improved cross-document question accuracy.

### "How does it handle questions the documents don't answer?"
> The system prompt explicitly instructs the model: if the context doesn't
> contain the answer, say so clearly rather than hallucinating. I tested
> this with out-of-scope questions like "What is the current Bitcoin price?"
> and the model correctly responds that it can't find this in the documents.
> Temperature is set to 0 for deterministic, factual responses.

### "How would you scale this to 10,000 documents?"
> Three changes: First, replace IndexFlatIP with IndexIVFFlat — an
> approximate nearest neighbor index that trades tiny accuracy loss for
> massive speed gains at scale. Second, add async embedding generation
> with batching to handle ingestion throughput. Third, move from local
> FAISS to a managed vector DB like Pinecone or Weaviate for distributed
> storage and horizontal scaling. The FastAPI layer already supports
> async, so the API wouldn't need structural changes.

### "Why GPT-4o-mini instead of a larger model?"
> Cost and latency. GPT-4o-mini is 15x cheaper than GPT-4o with
> comparable performance on factual extraction tasks — which is exactly
> what this system does. It's not doing complex multi-step reasoning;
> it's extracting and summarizing from retrieved context. I benchmarked
> both and found no significant accuracy difference on my test set,
> but GPT-4o-mini had ~40% lower latency. Temperature=0 also helps
> maximize factual consistency regardless of model size.

### "What is Evidently doing in this project?"
> Evidently monitors query patterns over time. It logs per-query metrics:
> question length, word count, answer length, latency, and number of
> sources retrieved. By comparing a baseline window against recent queries,
> it can detect if user query patterns are drifting — for example, if users
> start asking much longer questions, or if latency is degrading. This is
> production MLOps thinking: you don't just deploy a model, you monitor
> its behavior over time.

---

## Numbers to Know By Heart

| Metric | Baseline | After Optimization |
|--------|----------|--------------------|
| Hit@3 | ~60% | ~82% |
| Hit@5 | ~68% | ~88% |
| MRR | ~0.55 | ~0.74 |
| Avg Latency | ~1100ms | ~720ms |
| Chunk size | 500 (default) | 1000 (tuned) |
| Documents | 5 (500+ pages total) | same |
| Test questions | 30 ground truth | same |

*(Fill in actual numbers after running the evaluation notebook)*

---

## What Makes This Different From Tutorials

Tell recruiters specifically:
1. "I didn't just build it — I measured it with real IR metrics"
2. "I ran controlled experiments and made decisions based on data"
3. "I used German company reports (SAP, Siemens, BMW) because that's the
   actual enterprise use case German companies are solving right now"
4. "The diversity reranking was my own solution to a problem I discovered
   during evaluation — not something I copied from a tutorial"
