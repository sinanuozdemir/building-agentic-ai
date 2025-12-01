from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os

def get_structured_scorer(model_name: str = "meta-llama/llama-4-scout", temperature: float = 0):
    """
    Returns a function that scores an AI response against ground truth using a specified model.
    """
    # Define the Pydantic model for structured output
    class ScoreResponse(BaseModel):
        reasoning: str = Field(description="The reasoning process of what score you should pick. This should be detailed. Outline the points in the ground truth that the AI response correctly answered and incorrectly missed.")
        score: int = Field(description="An integer between 0 and 3 representing the correctness of the AI response compared to the ground truth")

    # Create a prompt template for scoring AI responses against ground truth
    score_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert evaluator. "
                "Given an AI's response to a question and the ground truth answer, "
                "score the AI's response on a scale from 0 to 3 based on correctness:\n"
                "0 = Completely does not match the ground truth or is irrelevant\n"
                "1 = Partially matches the ground truth, but with major errors or omissions\n"
                "2 = Mostly matches the ground truth, but with at most a single minor error or at most a single missing detail\n"
                "3 = Completely matches the ground truth exactly"
            )
        ),
        (
            "human",
            (
                "Question: {question}\n"
                "Ground Truth: {ground_truth}\n\n"
                "AI Response: {ai_response}\n"
                "The goal is to match the AI response to the ground truth. Your score should be based on how well the AI response matches the ground truth, nothing else. Score the AI response from 0 to 3."
            )
        )
    ])

    llm = ChatOpenAI(
        model=model_name, temperature=temperature, 
        base_url="https://openrouter.ai/api/v1", 
        api_key=os.getenv("OPENROUTER_API_KEY"),
        extra_body={"provider": {"order": ['groq']}}
        )
    structured_llm = llm.with_structured_output(ScoreResponse)

    def score_ai_response(question: str, ai_response: str, ground_truth: str):
        prompt = score_prompt.format(
            question=question,
            ai_response=ai_response,
            ground_truth=ground_truth
        )
        return structured_llm.invoke(prompt)

    return score_ai_response
from pydantic import BaseModel, Field

# Define the Pydantic model for structured output
class ScoreResponse(BaseModel):
    reasoning: str = Field(description="The reasoning process of what score you should pick")
    score: int = Field(description="An integer between 0 and 3 representing the correctness of the AI response compared to the ground truth")

# Create a prompt template for scoring AI responses against ground truth
score_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert evaluator. "
            "Given an AI's response to a question and the ground truth answer, "
            "score the AI's response on a scale from 0 to 3 based on correctness:\n"
            "0 = Completely incorrect or irrelevant\n"
            "1 = Partially correct, but with major errors or omissions\n"
            "2 = Mostly correct, but with minor errors or missing details\n"
            "3 = Completely correct and matches the ground truth"
        )
    ),
    (
        "human",
        (
            "Question: {question}\n"
            "AI Response: {ai_response}\n"
            "Ground Truth: {ground_truth}\n\n"
            "Score the AI response from 0 to 3."
        )
    )
])