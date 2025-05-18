from langchain.utils import TavilyClient
from openai import OpenAI
import os
import json
import re
from state_definitions import InvestmentState  # 중앙 집중식 상태 정의에서 가져옴

# API 키 설정
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 클라이언트 초기화
tavily = TavilyClient(api_key=TAVILY_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

def extract_company_name(title: str) -> str:
    # 쉼표(,) 또는 첫 공백을 기준으로 회사명만 잘라냄
    return title.split(",")[0].split(" ")[0].strip()

def competitor_analysis(state: InvestmentState) -> InvestmentState:
    """
    경쟁사 분석 에이전트 - 스타트업 도메인의 경쟁사를 분석하고 차별성 점수를 매깁니다.
    """
    # 스타트업 정보 추출 (이전 에이전트에서 설정한 값)
    domain = state.get("startup_info", {}).get("domain", "헬스케어")
    startup_name = state.get("startup_info", {}).get("name", "")
    
    # 1. Tavily 검색
    query = f"{domain} 스타트업 경쟁사 분석 {startup_name} 차별성"
    result = tavily.search(query=query, search_depth="advanced")
    documents = result.get("results", [])[:3]  # 상위 3개 문서만 가져옴
    
    # 2. 문서 텍스트 정리
    document_text = "\n\n".join([f"{doc['title']}\n{doc['content']}" for doc in documents])
    
    # 3. GPT로 경쟁우위 점수 요청
    system_prompt = (
        "당신은 스타트업 투자 전문가입니다. 아래 경쟁사 정보들을 바탕으로 다음을 평가하세요:\n"
        f"- {startup_name or '해당 도메인의 스타트업'}이 경쟁사보다 차별성이 얼마나 뚜렷한지 0~10점으로 평가\n"
        "- 해당 점수의 이유 설명\n"
        "- 주요 경쟁사 목록과 각 경쟁사의 강점/약점\n"
        "출력은 다음 JSON 형식으로 작성하세요:\n"
        "{\n"
        '  "competitive_score": float,\n'
        '  "competitive_reasoning": str,\n'
        '  "competitors": [\n'
        '    {"name": str, "strengths": [str], "weaknesses": [str]}\n'
        '  ]\n'
        "}"
    )
    
    user_prompt = f"도메인: {domain}\n스타트업: {startup_name}\n\n경쟁사 관련 정보:\n{document_text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",  # 최신 모델 사용
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
        )
        
        # 4. GPT 응답 파싱
        raw_output = response.choices[0].message.content
        
        # JSON 형식 추출 (정규식 사용)
        json_match = re.search(r"\{[\s\S]*\}", raw_output)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        else:
            # JSON 형식이 아닌 경우 기본값 설정
            parsed = {
                "competitive_score": 5.0,
                "competitive_reasoning": "경쟁사 분석 결과를 파싱할 수 없습니다.",
                "competitors": []
            }
            
    except Exception as e:
        print(f"경쟁사 분석 중 오류 발생: {str(e)}")
        # 오류 발생 시 기본값 설정
        parsed = {
            "competitive_score": 5.0,
            "competitive_reasoning": f"경쟁사 분석 중 오류 발생: {str(e)}",
            "competitors": []
        }
    
    # 5. 경쟁사 문서 객체 생성
    competitor_docs = []
    for doc in documents:
        competitor_docs.append({
            "title": extract_company_name(doc.get("title", "")),
            "url": doc.get("url", ""),
            "content": doc.get("content", "")
        })
    
    # 6. 경쟁사 분석 결과 저장
    state["competitors"] = parsed.get("competitors", [])
    
    # 7. 시장 분석 정보 업데이트
    if "market_analysis" not in state:
        state["market_analysis"] = {}
    
    state["market_analysis"].update({
        "competitor_documents": competitor_docs,
        "competitive_score": parsed.get("competitive_score", 5.0),
        "competitive_reasoning": parsed.get("competitive_reasoning", "")
    })
    
    # 8. 상태 업데이트
    state["status"] = "competitor_analysis_completed"
    
    return state

# 랭그래프 노드 생성 함수 - 메인에서 임포트할 때 사용
def create_competitor_analysis_agent():
    from langgraph.graph import StateGraph
    
    graph = StateGraph(InvestmentState)
    graph.add_node("competitor_analysis", competitor_analysis)
    
    # 시작점과 종료점이 같은 단일 노드 그래프
    graph.set_entry_point("competitor_analysis")
    graph.set_finish_point("competitor_analysis")
    
    return graph.compile()

# 단독 실행 테스트용
if __name__ == "__main__":
    from state_definitions import InvestmentState
    
    # 테스트용 초기 상태 생성
    test_state = {
        "startup_info": {
            "name": "MediHealth AI",
            "domain": "의료 AI 진단"
        },
        "competitors": [],
        "market_analysis": {},
        "investment_recommendation": {},
        "report_data": {},
        "status": "starting"
    }
    
    # 에이전트 실행
    result_state = competitor_analysis(test_state)
    
    # 결과 출력
    print(f"경쟁력 점수: {result_state['market_analysis']['competitive_score']}")
    print(f"경쟁력 분석: {result_state['market_analysis']['competitive_reasoning']}")
    print(f"경쟁사 수: {len(result_state['competitors'])}")