# Chapter 13: SFT vs. RAG — When to Bake Knowledge In vs. Retrieve It

On March 12, 2024, a financial compliance assistant deployed by a mid-sized investment advisory firm gave an analyst a number: $15,000. That was the SEC's penalty threshold for late Form ADV filings, accurate as of the model's training cutoff in November 2023. The analyst used it to assess a client's exposure and file a response brief. The actual threshold, following a rule amendment effective February 1, 2024, was $25,000. The discrepancy was not discovered until the client received a notice of deficiency six weeks later.

The model had not hallucinated. It had remembered correctly. That is what made the failure so difficult to detect and so worth understanding. The system returned a confident, internally consistent, syntactically impeccable answer that happened to describe a world that no longer existed.

This kind of failure does not come from a bad model. It comes from a mismatch between where knowledge is stored and how often that knowledge changes. Understanding that mismatch and designing against it before it ships is what this chapter is about.

The SFT/RAG decision is a useful entry point into a framework that will recur throughout this book: the structure–processing–performance tetrahedron. The four vertices are structure, which is where knowledge is stored and how the system is organized; processing, which is how queries are routed, retrieved, and generated at inference time; performance, which covers latency, cost, and accuracy as measured outputs; and failure visibility, which is whether the system's failure modes are detectable before they cause harm. Every architectural decision in an AI system moves value between these four vertices simultaneously. You cannot improve one without accepting a tradeoff against at least one other. The SFT/RAG routing decision is a clean example: encoding knowledge in weights improves inference latency at the cost of making staleness failures invisible to standard evaluation. Routing to retrieval restores failure visibility and freshness but adds latency and token cost. The tetrahedron does not tell you which tradeoff to make. It tells you that you are always making one, and that the engineering discipline is in naming it before deployment rather than discovering it from a deficiency notice six weeks later.

---

## The Two Places Knowledge Lives

Every piece of information that shapes a language model's output lives in one of two places: the weights of the network, or the context window of the current inference call. These are not equivalent storage media. They have different write speeds, different read speeds, and critically different failure modes. Treating them as interchangeable is the architectural error that produced the Chicago incident.

The most useful analogy here is embedded systems memory. Firmware engineers distinguish between flash storage, which is fast to read, expensive to update, and unchangeable at runtime, and RAM, which can be written and overwritten at any point in the process lifecycle. A sensor reading belongs in RAM. A firmware version belongs in flash. Loading sensor readings into flash produces a system that cannot respond to change. Loading firmware into RAM produces a system that is unstable. The same logic, applied to AI systems, defines the SFT/RAG decision.

Supervised fine-tuning is flash. When you fine-tune a language model, you run a gradient-descent optimization that adjusts the network's parameters so its probability distribution over output tokens shifts toward the patterns in your training data. The information is not stored as retrievable records. It is compressed into the geometry of a high-dimensional function. A 7B-parameter model completes an inference call in roughly 50–200ms on modern hardware with no external I/O, because the knowledge is already in the weights and access costs only floating-point operations. But updating those weights requires retraining: days of GPU compute at minimum, plus evaluation and deployment. The effective write latency for parametric memory is measured in weeks.

Retrieval-augmented generation is RAM. The model weights stay frozen. At query time, the system retrieves document chunks from an external store, typically a vector database indexed by semantic embedding, and injects them into the context window before generation. The knowledge lives in the document store, not the weights. Update a document and every subsequent query sees the change. When the SEC amended its penalty thresholds on February 1, 2024, a well-maintained RAG system needed only a document ingestion run to reflect the new figure. The model sees the correct number because it was handed the correct document.

The access mechanism matters as much as the storage location. In a RAG pipeline, the retrieval step is not a lookup. It is a ranked similarity search over a high-dimensional embedding space. The chunk returned is the chunk whose vector representation is closest to the query's vector representation, not necessarily the chunk that contains the correct answer. A query phrased differently from the source document may retrieve a plausible but irrelevant chunk, or miss the correct one entirely. The processing step introduces a failure mode that the storage decision alone does not determine.

The cost is latency and token budget. A RAG pipeline adds at minimum two I/O operations to every inference call: embedding the query and retrieving from the vector store. On a cloud-hosted store with sub-100ms p99 latency, this adds roughly 120–200ms and approximately 1,500 tokens of retrieved context per request. At mid-2024 GPT-4-class pricing of around $0.01 per thousand input tokens, three 512-token chunks cost $0.015 per query in retrieval context alone, before the query or response are counted. At 100,000 queries per day, that is $1,500 daily on retrieval context.

That figure is the retrieval context line only. The full per-query cost has four components, and collapsing them into a single number obscures the tradeoffs. At 10,000 queries per day on a two-tier hybrid system, the breakdown looks like this.

Query embedding: the retrieval pipeline must embed each incoming query before searching the vector store. At mid-2024 embedding API pricing of approximately $0.0001 per 1,000 tokens, a 50-token query costs roughly $0.000005. Negligible individually; $0.05 per day at 10,000 queries.

Vector store retrieval: cloud-hosted approximate nearest-neighbor search at sub-100ms p99 runs approximately $0.002 per query on managed infrastructure, covering compute and I/O. At 10,000 queries per day, that is $20 per day.

Retrieved context tokens: three 512-token chunks at $0.01 per 1,000 input tokens costs $0.015 per RAG query. If 30% of queries are routed to RAG, the high-V fraction, this is $0.015 multiplied by 3,000 queries, giving $45 per day.

SFT inference: the remaining 70% of queries hit the fine-tuned backbone directly, with no retrieval overhead. At $0.005 per query for a 7B-parameter hosted model, that is $35 per day.

Total at 10,000 queries per day: approximately $100 per day, split roughly evenly between retrieval infrastructure and inference. The naive approach of routing everything to RAG at the same volume costs approximately $185 per day, an 85% premium for routing stable knowledge through a retrieval pipeline it does not need. The volatility-indexed hybrid recovers that margin while eliminating the staleness failure mode for high-V categories.

The decision criterion follows directly from the mechanism. Define the knowledge volatility index V for a category of facts as:

$V = \frac{1}{T_{\text{update}}}$

where $T_{\text{update}}$ is the expected interval between updates to the ground truth of those facts, expressed as a multiple of your model retraining cycle. If your fine-tuning cycle is quarterly and your knowledge updates daily, $T_{\text{update}}$ is 1/90 of your retraining period and V equals 90. High-V knowledge belongs in retrieval. Low-V knowledge, where $T_{\text{update}}$ is much greater than $T_{\text{retrain}}$, can be encoded in weights without staleness risk. The routing decision follows from the number, not from intuition.


<img width="944" height="768" alt="Screenshot 2026-04-04 120632" src="https://github.com/user-attachments/assets/77723aab-656b-496d-904b-f5ee701b4eee" />



Working through the classification is worth doing once in full, because the table below presents conclusions rather than the reasoning that produces them. Take the first row. SEC penalty thresholds are revised by rulemaking, and the SEC's rulemaking cadence for civil monetary penalties runs roughly quarterly. The firm's retraining cycle is biannual. So $T_{\text{update}}$ is 0.25 years and $T_{\text{retrain}}$ is 0.5 years, giving $T_{\text{update}}$ expressed as a multiple of the retraining cycle: 0.25 divided by 0.5, which equals 0.5. Therefore V equals 1 divided by 0.5, giving 2.0 at minimum. Because penalty schedules can be amended mid-cycle without notice, the effective V is higher. We round to 4.0 and route to RAG.

Now take row five: Investment Adviser Act definitions. The statutory text of the 1940 Act has not been substantively amended in decades. A conservative estimate of $T_{\text{update}}$ is twenty years, giving $T_{\text{update}}$ divided by $T_{\text{retrain}}$ equal to 40, and V equal to 0.025. At any reasonable retraining cadence, this knowledge will be current in the weights when the model ships and will still be current when the model is retired. Route to SFT.

The decision boundary is not a bright line but a risk threshold. A reasonable default: any knowledge category with V greater than 1, meaning it updates faster than your retraining cycle, belongs in retrieval unless you can demonstrate that the cost of a stale answer is acceptable. The Chicago case is the calibration point for what acceptable means.

Here is how the Chicago team should have applied this before deployment. Their retraining cycle was biannual. The compliance assistant needed to answer questions across a realistic cross-section of investment advisory knowledge:


<img width="1257" height="500" alt="Screenshot 2026-04-04 121202" src="https://github.com/user-attachments/assets/074e6a66-fc31-42bb-a25d-c044676d140a" />




The $15,000 figure lived in row one. The team encoded it in the storage medium appropriate for rows four through six. That mismatch, V equal to 4 knowledge in a system whose write latency is 0.5 years, is the architectural error, stated as a number.

The table also shows something the intuitive framing obscures: most of the firm's knowledge is low-V. Statutory definitions, fiduciary standards, filing structures are stable over years and are correctly served by parametric memory. The right architecture is not to use RAG for everything to be safe. It is to route by V, accept the retrieval cost only where volatility justifies it. A system that routes all six categories to RAG wastes approximately $0.015 per query on knowledge the model already encodes correctly. At 100,000 queries per day, that is $1,500 in daily retrieval overhead on rows four through six alone. A system that routes only the top two rows to RAG eliminates the failure mode that produced the deficiency notice and recovers most of that cost.

---

## Where the Simple Model Breaks

The volatility framework above is correct as far as it goes. It does not go far enough, and leaving a student with only this model would be doing them a disservice.

The parametric memory model assumes that fine-tuned weights reliably store whatever facts were in the training data. They do not. Empirical studies of factual recall in transformer models consistently show that models trained on a fact appearing once or twice in a corpus recall it with lower confidence and higher error rate than models trained on that fact repeatedly, and that recall degrades in ways that produce no warning signal. The model does not say it is uncertain about this because it saw it only twice. It says $15,000 with the same fluency and token probability as it says water is wet. Fine-tuning teaches a model how to reason about a domain, covering vocabulary, citation structure, and response format, far more reliably than it stores specific numerical values. Point-lookup facts belong in retrieval even when their volatility index is low, because the storage medium is unreliable for point-lookup tasks regardless of volatility.

The contextual memory model has its own failure mode at production scale. A vector store that has been running for more than six months without explicit hygiene procedures is not a consistent database. It is a document accumulation. Consider a compliance system ingesting regulatory updates from three sources: SEC rulemaking releases, FINRA guidance notices, and internal policy documents. Each source has a different ingestion pipeline, a different embedding model version, and a different update frequency. If a regulation is updated and the primary ingestion pipeline processes the new document but the deduplication check fails to deprecate the old chunk, the vector store now contains two chunks representing the same regulation at different effective dates. A retrieval call that returns both hands the model contradictory context. The model will synthesize, and the synthesis may be wrong in ways that are undetectable without source-level auditing.

The processing failure here is distinct from the storage failure. The documents may be correct and current. The retrieval mechanism may still return the wrong one, not because the store is stale, but because two chunks representing the same regulation at different effective dates score similarly against the query vector, and the ranking function has no way to prefer the more recent without explicit metadata filtering. Correct storage plus flawed processing still produces a wrong answer.


<img width="1160" height="884" alt="Screenshot 2026-04-04 121826" src="https://github.com/user-attachments/assets/77b333c1-3852-458f-8207-22a18c2e6cc6" />



RAG does not eliminate the consistency problem. It relocates it from the model weights to the data infrastructure. This is often a better place to have the problem, because data infrastructure consistency is a more mature engineering discipline than model weight management. But it is not a solved problem, and architects who deploy RAG because it sounds grounded without building explicit vector store hygiene procedures are trading one silent failure mode for another.

---

## The Failure That Was Predictable

Return to Chicago. The causal chain that produced the liability is worth tracing link by link, because each link represents a design decision that could have broken the chain before the harm materialized.

The firm's engineering team chose to fine-tune their model on the complete regulatory corpus, including volatile penalty schedules, to achieve sub-200ms latency on compliance queries. This was Link 1: a structural decision to encode high-V knowledge in parametric memory. The volatility index for SEC penalty thresholds is approximately 4 annually. The retraining cycle was biannual. Encoding quarterly-changing facts into a system with a six-month write latency is a mismatch of V equal to 4 against a storage medium whose effective update period is 0.5 years.

The firm had a documentation update process that modified the internal compliance wiki when regulations changed. This process was not connected to the model retraining pipeline. The wiki and the weights were maintained independently, by different teams, with different schedules. This was Link 2: an organizational gap that made the architectural mismatch invisible in normal operations.

The system logged query-response pairs but ran no ground-truth comparison for numerical claims. There was no eval suite that tested the model's numerical outputs against a live regulatory feed. This was Link 3: a monitoring gap that ensured the failure would accumulate silently rather than trigger an alert.

The model produced answers with no calibrated uncertainty signal for the figures it cited. The figure $15,000 was returned with the same fluency and formatting as every other output. This was Link 4: an output design that gave the analyst no reason to verify the number.

The analyst submitted a response brief based on the incorrect threshold. The brief was filed with a third party and became irreversible. This was Link 5: harm materialization.

Every link is a correctable design decision. The architecture could have routed penalty thresholds to RAG. The update process could have been connected to an ingestion pipeline. The monitoring could have compared numerical outputs to a regulatory API. The output formatting could have included a staleness timestamp on numerical claims. None of these corrections were made, because the team did not trace the failure chain from architecture to harm before deployment.

There is one more detail worth knowing. The architectural review records show that a senior engineer raised the staleness concern during the design phase and proposed a hybrid approach, with RAG for numerical thresholds and SFT for reasoning style. The proposal was rejected on latency grounds: adding a retrieval step would push p99 latency above the 200ms SLA. This was a technically defensible decision under the latency constraint and an architecturally indefensible one under the risk constraint. The engineers who made it did not have a framework for making the tradeoff explicit. What follows is that framework.

---

## Designing the Hybrid System

The binary choice between SFT and RAG is a false dichotomy in any production system with heterogeneous knowledge. The practical architecture is a router that classifies each query by volatility index and directs it to the appropriate backend.

The router is itself a model, a classifier that predicts whether a given query requires current information or is adequately served by parametric knowledge. Queries classified as high-V are routed to the RAG pipeline. Queries classified as low-V are served by the fine-tuned backbone directly. This separation captures the latency advantage of parametric memory for stable knowledge while preserving retrieval freshness for volatile facts.

The processing path for each route is worth tracing explicitly. A low-V query routed to the SFT backbone travels one path: tokenization, forward pass through the fine-tuned weights, detokenization. No external calls, no ranking, no context injection. A high-V query routed to RAG travels a longer path: query embedding, approximate nearest-neighbor search over the vector store, chunk retrieval, context assembly, forward pass with the injected context, detokenization. Each additional step is a point where the processing mechanism can introduce error independently of whether the underlying knowledge is correct.

The routing decision for the firm's compliance system is illustrative. A query asking for the standard definition of a 10-K filing is a low-V query: the definition has not changed meaningfully in decades, the model encodes it reliably, and routing it to RAG wastes tokens and adds latency. A query asking for the penalty limits for Q3 2024 late ADV filings is a high-V query: the answer is a specific numerical value tied to a specific regulatory period, the correct answer changes over time, and the only reliable source is a live document. The router does not need to understand the content of the query in depth. It needs to predict whether the answer is time-stable or time-sensitive, which is a classification task well within the capability of a lightweight encoder model.

The router's false negative rate, the fraction of high-V queries it misroutes to the SFT backbone, is the single most important reliability parameter in this architecture. A lightweight encoder classifier on this task typically achieves a false negative rate of 2 to 5 percent without extensive tuning. At 2 percent, a system processing 100,000 queries per day delivers roughly 2,000 stale, unverifiable answers before any monitoring cycle catches them. At 5 percent, that figure is 5,000. In a generic search application, a 2 to 5 percent error rate is acceptable noise. In a compliance system where each misrouted query is a potential liability event, it is a structural risk that must be measured continuously, not estimated once at deployment.

This rate cannot be measured without a live ground-truth feed, a pipeline that compares the model's numerical outputs against current regulatory values at query time. Building that feed is not an optional enhancement. It is part of the architecture, and its cost belongs in the system design budget, not the operations runbook.

What minimum monitoring infrastructure means in practice is determined by where the ground-truth feed itself can fail. Consider the feed latency blind spot. A regulatory API that supplies ground truth updates on a 24-hour batch cycle creates a structural gap: if a rule amendment is published at 11pm, the batch runs at 2am, and the comparison pipeline runs at 6am, every query answered between 11pm and 6am is compared against yesterday's ground truth. The monitoring infrastructure is running. It is simply running against stale data. The system passes those queries as correct, and they may not be.

This is not a monitoring bug. It is a monitoring architecture decision with a known failure mode. A system designed to detect staleness within 24 hours must account for the full latency chain: source publication delay, feed ingestion delay, comparison pipeline execution delay, and alert propagation delay. Each link has a latency budget. The sum of those budgets is the actual detection window, not the 24-hour target specified in the requirements. Designing monitoring infrastructure means specifying each link and its latency, not just asserting that a comparison pipeline exists.

A further refinement available in latency-constrained systems is RAG result caching. Retrieved chunks for high-frequency regulatory queries, such as the penalty schedules that analysts check dozens of times per day, can be cached at the application layer with a TTL that matches the update frequency of the underlying regulatory source. A query about current ADV penalty thresholds that hits a warm cache returns in under 50ms. A cache miss triggers a full retrieval. This pattern recovers most of the latency cost of RAG for stable-but-volatile knowledge, and it is the architecture that could have satisfied both the 200ms SLA and the freshness requirement in the Chicago case, had the tradeoff been made explicit. The cache introduces its own failure mode: a TTL set longer than the source update frequency will serve stale cached chunks with the same confidence as a fresh retrieval. That failure mode requires its own monitoring layer, with cache invalidation events logged and compared against source update events, and an alert threshold on the gap.


<img width="1017" height="601" alt="Screenshot 2026-04-04 121009" src="https://github.com/user-attachments/assets/f3bb1f44-1afd-4769-b817-29855e16514e" />


## Structure, Processing, and What Evaluations Cannot See

The SFT/RAG decision connects to all four vertices of the structure–processing–performance tetrahedron introduced at the start of this chapter, but the connection most often missed is the one between structure and failure mode visibility.

The Chicago system was evaluated extensively before deployment. It passed. The evaluations ran on the training distribution, regulatory text and compliance scenarios as they existed at the model's training cutoff. The structural mismatch between the update frequency of penalty schedules and the write latency of parametric memory was invisible to any evaluation that did not include post-cutoff ground truth, because no post-cutoff ground truth existed at the time of evaluation. The system passed every test it was given, and still shipped a latent failure.

This is the generalized lesson. Evaluations test a system against what it was designed to handle. Architecture determines what the system was designed to handle. If the architecture is wrong, if it encodes volatile facts in immutable memory, then the failures it will eventually produce are not in the distribution the evaluations cover. They will appear after deployment, in conditions the evaluation suite was not designed to simulate.

The implication for system design is that the monitoring infrastructure is architecturally load-bearing, not operationally supplementary. A compliance system without a live ground-truth comparison pipeline for numerical claims is not a deployed system with incomplete monitoring. It is a system with a structural gap that will produce undetected failures at a rate determined by the volatility of the knowledge it encodes and the length of its retraining cycle. The engineering budget for that monitoring pipeline belongs in the architecture document, not the operations runbook.

---

## Problems

**13.1** A legal research platform indexes case law and statute text. For each of the following knowledge categories, estimate a qualitative volatility index relative to a 90-day fine-tuning cycle and recommend SFT, RAG, or a hybrid approach. Justify each recommendation by naming the failure mode that the wrong choice would produce, not just the correct choice.

a. The text of a statute as enacted in 1978.
b. The current citation count for a landmark Supreme Court case.
c. The definition of tortious interference as established in common law.
d. The effective date of a regulatory amendment published last month.
e. The standard structure of a contract dispute brief.

**13.2** A RAG system for a pharmaceutical firm retrieves drug interaction data from a clinical database updated nightly. The embedding model used to index the database was trained in 2022. In 2024, the FDA updated the terminology used to classify drug-drug interactions. The new terminology uses different noun phrases than the old, but refers to the same underlying clinical concepts.

Describe precisely what happens to retrieval quality for queries phrased in the new terminology. Identify the link in the causal chain at which the failure originates. Then propose a monitoring strategy that would detect the failure before it reaches patient-facing outputs. What ground truth is required to implement that strategy, and what does acquiring it cost?

**13.3** You are designing the knowledge management layer for a medical decision support system used by emergency physicians. The system must provide drug dosage recommendations, differential diagnosis support, and institutional protocol lookups. Hard constraints: p99 latency under 300ms, all factual claims auditable to a source document. The knowledge base includes both stable pharmacological principles, which change on a timescale of years, and institutional protocols updated monthly by the hospital's pharmacy committee.

Specify what is fine-tuned and what is retrieved, the routing logic and its classification target, the failure mode of the router's false negative path, and the minimum monitoring infrastructure needed to detect that failure within 24 hours. Include a cost estimate per query at 10,000 queries per day, broken down by component.

**13.4** The original hybrid proposal for the Chicago system, RAG for numerical thresholds and SFT for reasoning style, was rejected because adding a retrieval step would push p99 latency above the 200ms SLA. Using the caching strategy described in this chapter, construct a revised architecture that satisfies both the latency constraint and the freshness requirement. Provide a quantitative argument showing how your architecture achieves sub-200ms p99 latency under realistic cache hit rate assumptions. Then identify the new failure mode your caching layer introduces, and describe the monitoring required to detect it.

Note that there is no architecture without failure modes. The question is which failure modes you are equipped to defend against.

**13.5** You are building a customer support AI for a software company with a rapidly evolving product: major releases every six weeks, patch releases several times per week. Support queries span how-to questions, which are relatively stable, and questions about specific feature behavior, which are version-specific and change with every release. The latency requirement is 1,500ms p99. Your team has two engineers, neither of whom has operated a vector store at production scale.

Design a phased architecture that is deployable in the first two weeks with available resources and evolvable toward a mature hybrid system over six months. For each phase, name the architectural property you are trading away and the failure mode you are accepting in exchange for speed of deployment. Explain why that tradeoff is defensible at the initial deployment scale but not at the six-month target.

---

## A Note on AI-Assisted Architectural Review

After completing Problem 13.3, submit your architecture to an AI assistant with the following instruction: generate five adversarial test cases, meaning specific query types or operational scenarios, that would expose failure modes in the design.

Use the AI to generate the cases. The generation step, producing diverse adversarial scenarios across the full design space, is where a capable AI assistant saves you significant time. What you must do yourself is evaluate each case: determine whether the predicted failure is accurate, whether the causal explanation is correct, and whether your architecture actually handles the scenario in a way the AI did not recognize. In at least one case, the AI will mischaracterize the failure mode, misattributing it to the wrong component, or generating a scenario your design handles correctly without realizing it. Identify that case, explain precisely where the AI's reasoning broke down, and document your correction.

What you are practicing here is not AI use. It is the judgment that AI cannot supply: knowing when a structurally confident answer is wrong, and being able to say specifically why. That judgment is what the rest of this chapter was building toward.

---

*This chapter assumes familiarity with transformer inference covered in Chapter 11 and vector database indexing covered in Chapter 12. Chapter 14 extends the routing architecture introduced here to multi-agent orchestration.*
