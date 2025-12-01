from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os


class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer.

Example 1:
Objective: Write a one-pager on the benefits of AI.
Plan:
1. Look up the latest news on AI on reputable sources.
2. For any sources that seem relevant, read the article and summarize the key points.
Final Result: the one pager in markdown format.

Example 2:
Objective: Write a tweet about the latest hurricane in the US.
Plan:
1. Look up the latest news on hurricanes in the US on reputable sources.
2. For any sources that seem relevant, read the article and summarize the key points.
Final Result: the tweet

The last step should NOT be the final result, the plan is only for the work it will take to get enough information to write the final result.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="openai/gpt-4.1", temperature=0.2, base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
).with_structured_output(Plan)


from typing import Union


class FinalResponseReady(BaseModel):
    """Explain in a sentence why you think the final answer is ready to be sent to the user."""
    response: str


class Act(BaseModel):
    """Action to perform. Either a final response to the user's objective or a plan to execute."""
    action: Union[FinalResponseReady, Plan] = Field(
        description="Action to perform. If you want to respond to user, use FinalResponse. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer.

Example 1:
Objective: Write a one-pager on the benefits of AI.
Plan:
1. Look up the latest news on AI on reputable sources.
2. For any sources that seem relevant, read the article and summarize the key points.
Final Result: the one pager in markdown format.

Example 2:
Objective: Write a tweet about the latest hurricane in the US.
Plan:
1. Look up the latest news on hurricanes in the US on reputable sources.
2. For any sources that seem relevant, read the article and summarize the key points.
Final Result: the tweet

The last step should NOT be the final result, the plan is only for the work it will take to get enough information to write the final result.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the following steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with an action of type Response."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="google/gemini-2.5-flash-lite-preview-06-17", temperature=0.3, base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
).with_structured_output(Act)
