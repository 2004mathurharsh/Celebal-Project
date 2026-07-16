import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

class PatchContextRAG:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in .env file.")
            
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=self.api_key
        )
        
        # Load FAISS index if it exists
        if os.path.exists(FAISS_INDEX_DIR):
            self.db = FAISS.load_local(
                FAISS_INDEX_DIR, 
                self.embeddings,
                allow_dangerous_deserialization=True # Required for loading local FAISS pickle file
            )
            print("FAISS index loaded successfully.")
        else:
            self.db = None
            print("WARNING: FAISS index not found. Please run ingest.py first.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0.0,
            google_api_key=self.api_key,
            max_retries=0
        )
        
    def retrieve(self, query, k=5):
        """Retrieves documents using FAISS with Maximal Marginal Relevance (MMR) for diversity."""
        if not self.db:
            print("FAISS database not initialized.")
            return []
            
        # k: number of documents to return, fetch_k: number of documents to pass to MMR algorithm
        retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20}
        )
        return retriever.invoke(query)
        
    def extract_citations_from_text(self, text):
        """Extracts citations like [PR #123], [Issue #456], or [Commit 1a2b3c4] from text."""
        # Match [PR #123], [Issue #123], [Commit 1a2b3c4] (hex SHA, 7 to 40 characters)
        pr_issue_pattern = r"\[(?:PR|Issue)\s*#\d+\]"
        commit_pattern = r"\[Commit\s*[a-f0-9]{7,40}\]"
        
        citations = re.findall(pr_issue_pattern, text, re.IGNORECASE) + re.findall(commit_pattern, text, re.IGNORECASE)
        # Standardize spaces and casing (e.g. [pr #123] -> [PR #123])
        standardized = []
        for c in citations:
            c_clean = c.strip("[]")
            parts = c_clean.split()
            if len(parts) == 2:
                label = parts[0].upper()
                id_val = parts[1]
                standardized.append(f"[{label} {id_val}]")
            else:
                standardized.append(c)
        return list(set(standardized))
        
    def check_fabricated_citations(self, response_text, retrieved_docs):
        """Checks if citations mentioned in response actually exist in retrieved docs."""
        allowed_citations = set()
        for doc in retrieved_docs:
            allowed_citations.add(doc.metadata.get("citation", "").upper())
            
        found_citations = self.extract_citations_from_text(response_text)
        fabricated = []
        
        for c in found_citations:
            c_upper = c.upper()
            if c_upper not in allowed_citations:
                # Double check with a flexible check
                clean_c = c_upper.replace("[", "").replace("]", "").replace("#", "")
                parts = clean_c.split()
                if len(parts) == 2:
                    val = parts[1]
                    matched = False
                    for allowed in allowed_citations:
                        if val in allowed:
                            matched = True
                            break
                    if not matched:
                        fabricated.append(c)
                else:
                    fabricated.append(c)
                    
        return fabricated, found_citations

    def run_nli_alignment_check(self, context_text, answer_text):
        """Performs LLM-based NLI check to verify if the context entails the answer's claims."""
        nli_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an NLI (Natural Language Inference) evaluator. Your task is to check if the generated answer is "
                "supported by the retrieved context. Specifically, analyze the statements in the Answer and verify if they "
                "are fully supported by the Context.\n\n"
                "Provide a structured assessment:\n"
                "1. Direct support: Identify which parts of the answer are supported by the context.\n"
                "2. Hallucinations/Unsupported claims: List any statement or claim in the answer not supported by the context.\n"
                "3. Verdict: Select one of: [FULLY SUPPORTED], [PARTIALLY SUPPORTED], or [NOT SUPPORTED].\n\n"
                "Keep your assessment very concise."
            )),
            ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
        ])
        
        nli_chain = nli_prompt | self.llm | StrOutputParser()
        assessment = nli_chain.invoke({"context": context_text, "answer": answer_text})
        
        verdict = "FULLY SUPPORTED"
        if "[PARTIALLY SUPPORTED]" in assessment:
            verdict = "PARTIALLY SUPPORTED"
        elif "[NOT SUPPORTED]" in assessment:
            verdict = "NOT SUPPORTED"
            
        return verdict, assessment

    def answer_query(self, query):
        """Orchestrates the retrieval, generation, citation verification, and NLI check."""
        if not self.db:
            return {
                "answer": "FAISS database not found. Please run the ingestion script first.",
                "citations": [],
                "retrieved_docs": [],
                "guard_status": {"passed": False, "reason": "No vector database"}
            }
            
        # Step 1: Retrieve context
        retrieved_docs = self.retrieve(query, k=5)
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found in the database.",
                "citations": [],
                "retrieved_docs": [],
                "guard_status": {"passed": True, "reason": "No relevant context"}
            }
            
        # Format the retrieved documents for context
        formatted_context = ""
        for i, doc in enumerate(retrieved_docs):
            formatted_context += f"--- Document {i+1} ---\n{doc.page_content}\n\n"
            
        # Step 2: Set up Chat Prompt for answering
        system_instructions = (
            "You are PatchContext, a RAG agent designed to answer questions about the design and architecture of the FastAPI framework.\n"
            "You are given retrieved commit history, pull request descriptions, and issue discussions.\n\n"
            "Rules for answering:\n"
            "1. Answer the query thoroughly, explaining the 'why' behind design decisions based strictly on the provided context.\n"
            "2. Ground your claims in the context. Every time you mention a fact from a specific commit, PR, or Issue, cite it in square brackets, e.g., '[Commit 1a2b3c4]', '[PR #123]', or '[Issue #456]'.\n"
            "3. Only use the citations present in the context. Do not invent commit SHAs, PR numbers, or Issue IDs.\n"
            "4. If the context does not contain enough information to answer the question, state that the context is insufficient, but provide whatever partial insights you can find. Do not make up answers.\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            ("human", "Context:\n{context}\n\nQuery: {query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Step 3: Initial Generation
        response = chain.invoke({"context": formatted_context, "query": query})
        
        # Step 4: Verification of citations (Hallucination Guard)
        fabricated, found = self.check_fabricated_citations(response, retrieved_docs)
        
        # Step 5: Self-correction if fabricated citations are found
        if fabricated:
            print(f"Hallucination Guard triggered! Fabricated citations found: {fabricated}. Initiating self-correction...")
            correction_prompt = ChatPromptTemplate.from_messages([
                ("system", system_instructions),
                ("human", (
                    "Context:\n{context}\n\n"
                    "Query: {query}\n\n"
                    "You previously generated this answer:\n{response}\n\n"
                    "CRITICAL ERROR: The following citations in your response are NOT present in the retrieved context: {fabricated_list}.\n"
                    "Please rewrite your answer. Ensure you ONLY cite sources that are explicitly listed in the Context. "
                    "If a citation cannot be verified, omit it and rely on the verified ones: {allowed_list}."
                ))
            ])
            
            allowed_list = ", ".join([d.metadata.get("citation", "") for d in retrieved_docs])
            correction_chain = correction_prompt | self.llm | StrOutputParser()
            
            response = correction_chain.invoke({
                "context": formatted_context,
                "query": query,
                "response": response,
                "fabricated_list": ", ".join(fabricated),
                "allowed_list": allowed_list
            })
            
            # Re-evaluate
            fabricated, found = self.check_fabricated_citations(response, retrieved_docs)
            
        # Step 6: Perform NLI entailment check on final answer
        nli_verdict, nli_assessment = self.run_nli_alignment_check(formatted_context, response)
        
        # Guard status
        guard_passed = (len(fabricated) == 0) and (nli_verdict != "NOT SUPPORTED")
        guard_status = {
            "passed": guard_passed,
            "fabricated_citations": fabricated,
            "nli_verdict": nli_verdict,
            "nli_assessment": nli_assessment
        }
        
        # Map citations to their actual URLs from retrieved documents
        citations_mapping = []
        for citation in found:
            if citation in fabricated:
                continue
                
            url = ""
            source_type = ""
            source_id = ""
            
            cit_clean = citation.upper().replace("[", "").replace("]", "").replace("#", "")
            for doc in retrieved_docs:
                doc_cit_clean = doc.metadata.get("citation", "").upper().replace("[", "").replace("]", "").replace("#", "")
                if cit_clean == doc_cit_clean or cit_clean in doc_cit_clean or doc_cit_clean in cit_clean:
                    url = doc.metadata.get("url", "")
                    source_type = doc.metadata.get("source_type", "")
                    source_id = doc.metadata.get("source_id", "")
                    break
                    
            citations_mapping.append({
                "label": citation,
                "url": url,
                "source_type": source_type,
                "source_id": source_id
            })
            
        return {
            "answer": response,
            "citations": citations_mapping,
            "retrieved_docs": retrieved_docs,
            "guard_status": guard_status
        }

if __name__ == "__main__":
    print("Testing PatchContextRAG with Gemini...")
    try:
        rag = PatchContextRAG()
        res = rag.answer_query("Why did we change or design APIRouter?")
        print("\n--- TEST RESPONSE ---")
        print(res["answer"])
        print("\nCitations:", res["citations"])
        print("\nGuard Status:", res["guard_status"])
    except Exception as e:
        print(f"Error testing RAG pipeline: {e}")
