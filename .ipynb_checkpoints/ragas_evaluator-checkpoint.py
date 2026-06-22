from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    
def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except Exception:
        return default

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    if not question or not answer:
        return {"response_relevancy": 0.0, "faithfulness": 0.0}
    
    # TODO: Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0))
    
    # TODO: Create evaluator_embeddings with model test-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model = "text-embedding-3-small"))
    
    # TODO: Define an instance for each metric to evaluate
    response_relevancy = ResponseRelevancy(llm = evaluator_llm, embeddings = evaluator_embeddings)
    faithfulness = Faithfulness(llm=evaluator_llm)
    metrics = [response_relevancy, faithfulness]
    
    # TODO: Evaluate the response using the metrics
    sample = {"user_input": question, "response": answer, "retrieved_contexts": contexts or []}
    dataset = Dataset.from_list([sample])
    result = evaluate(dataset = dataset, metrics = metrics, llm = evaluator_llm, embeddings = evaluator_embeddings, raise_exceptions = False, show_progress = False)
    scores = dict(result)
    
    # TODO: Return the evaluation results
    answer_relevancy = scores.get("answer_relevancy", 0.0)
    response_relevancy = safe_float(scores.get("response_relevancy", answer_relevancy))
    faithfulness = safe_float(scores.get("faithfulness", 0.0))
    return {"response_relevancy": response_relevancy, "faithfulness": faithfulness}
