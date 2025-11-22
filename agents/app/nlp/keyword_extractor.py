from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings


def build_keyword_chain():
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    system_prompt = (
        "You are an assistant specialized in extracting key information "
        "from a user's speech transcript. Your job is to:\n"
        "1. Extract 1-5 important **keywords** that capture the user's intent.\n"
        "2. Generate a short, clean **English prompt** that will be sent to a "
        "vision-language model (VLM) to help it understand the user's request.\n\n"
        "The output MUST be valid JSON in this exact format:\n"
        '{ "keywords": ["..."], "vlm_prompt": "..." }\n\n'
        f"Tone guideline: {settings.gentle_voice_style}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "User speech transcript: {transcript}"),
        ]
    )

    chain = prompt | llm
    return chain


def extract_keywords_and_prompt(transcript: str) -> Dict:
    """
    Call LLM to extract keywords from text and generate a prompt for VLM.
    Returns a dict: { "keywords": [...], "vlm_prompt": "..." }
    """
    chain = build_keyword_chain()
    resp = chain.invoke({"transcript": transcript})
    # LangChain msg.content is usually a string, try to parse as JSON
    import json

    try:
        data = json.loads(resp.content)
    except Exception:
        # If model doesn't return valid JSON, fall back to basic structure
        data = {"keywords": [], "vlm_prompt": resp.content}

    # Ensure basic fields exist
    if "keywords" not in data:
        data["keywords"] = []
    if "vlm_prompt" not in data:
        data["vlm_prompt"] = ""

    return data
