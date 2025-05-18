import os
import logging
from typing import Dict, Any
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langgraph.graph import StateGraph

# state_definitions.py에서 정의한 상태 가져오기
from state_definitions import InvestmentState, CandidateDocument

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 글로벌 변수로 데이터/모델 초기화
embedding_model = None  # 임베딩 모델
pinecone_index = None  # Pinecone 인덱스
openai_client = None  # OpenAI 클라이언트

# 초기화 함수
def init_resources(pinecone_api_key: str, openai_api_key: str, index_name: str):
    """필요한 리소스 초기화"""
    global embedding_model, pinecone_index, openai_client
    
    # OpenAI 설정
    openai_client = OpenAI(api_key=openai_api_key)
    
    # 임베딩 모델 로드
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Pinecone 클라이언트 초기화
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(index_name)
    
    logger.info(f"리소스 초기화 완료. 인덱스: {index_name}")

# 스타트업 탐색 함수
def startup_exploration(state: InvestmentState) -> InvestmentState:
    """
    스타트업 탐색 에이전트 - 투자 가능성 있는 스타트업을 검색하고 정보를 수집합니다.
    """
    # 사용자 쿼리 가져오기 (검색 키워드 또는 관심 도메인)
    user_query = state.get("user_query", "AI 스타트업")
    
    # 1. 관련 스타트업 검색 - Pinecone 벡터 검색 사용
    candidate_startup = search_related_startup(user_query)
    
    # 후보 스타트업이 없다면 기본 정보 설정
    if not candidate_startup:
        logger.warning(f"쿼리 '{user_query}'에 관련된 스타트업을 찾지 못했습니다.")
        
        # CandidateDocument 생성
        candidate_doc = CandidateDocument(
            user_query=user_query,
            name="관련 스타트업 없음",
            summary="검색 쿼리와 관련된 스타트업을 찾지 못했습니다.",
            domain="미정"
        )
        
        # 상태 업데이트
        state["startup_info"] = candidate_doc.dict()
        state["status"] = "startup_exploration_no_results"
        return state
    
    # 2. 스타트업 정보 수집 및 분석
    try:
        # Pinecone에서 찾은 스타트업 정보 활용
        startup_name = candidate_startup.get("name", "")
        startup_summary = candidate_startup.get("summary", "")
        startup_domain = candidate_startup.get("domain", "기술")
        
        # 3. CandidateDocument 생성
        candidate_doc = CandidateDocument(
            user_query=user_query,
            name=startup_name,
            summary=startup_summary,
            domain=startup_domain
        )
        
        # 4. 상태 업데이트
        state["startup_info"] = candidate_doc.dict()
        state["status"] = "startup_exploration_completed"
        
        logger.info(f"스타트업 '{startup_name}' 정보 수집 완료")
        
    except Exception as e:
        logger.error(f"스타트업 정보 수집 중 오류 발생: {str(e)}")
        
        # 오류 시 기본 정보로 CandidateDocument 생성
        candidate_doc = CandidateDocument(
            user_query=user_query,
            name=candidate_startup.get("name", "알 수 없음"),
            summary="정보 처리 중 오류가 발생했습니다.",
            domain=candidate_startup.get("domain", "기술")
        )
        
        # 상태 업데이트
        state["startup_info"] = candidate_doc.dict()
        state["status"] = "startup_exploration_error"
    
    return state

# 관련 스타트업 검색 함수
def search_related_startup(query: str) -> Dict[str, Any]:
    """Pinecone에서 쿼리와 관련된 스타트업 검색"""
    if embedding_model is None or pinecone_index is None:
        logger.error("리소스가 초기화되지 않았습니다. init_resources()를 먼저 호출하세요.")
        return {}
    
    try:
        # 쿼리 임베딩 생성
        query_embedding = embedding_model.encode(query).tolist()
        
        # Pinecone에서 가장 유사한 벡터 검색
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=1,  # 가장 유사한 1개만 가져오기
            include_metadata=True
        )
        
        # 결과가 있는지 확인
        if results.matches and len(results.matches) > 0:
            match = results.matches[0]
            
            # 메타데이터에서 필요한 정보 추출
            if match.metadata and "name" in match.metadata:
                startup_info = {
                    "name": match.metadata.get("name", "Unknown"),
                    "summary": match.metadata.get("summary", ""),
                    "domain": match.metadata.get("domain", "기술"),
                    "score": match.score  # 유사도 점수
                }
                
                # 도메인이 없으면 기본 설정
                if not startup_info["domain"] or startup_info["domain"] == "Unknown":
                    startup_info["domain"] = extract_domain(startup_info["name"], startup_info["summary"])
                
                return startup_info
        
        # 결과가 없으면 빈 딕셔너리 반환
        return {}
        
    except Exception as e:
        logger.error(f"스타트업 검색 중 오류 발생: {str(e)}")
        return {}

# 도메인 추출 함수
def extract_domain(name: str, summary: str) -> str:
    """텍스트 정보에서 스타트업 도메인 추출"""
    if openai_client is None:
        logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
        return "기술"  # 기본 도메인
    
    try:
        prompt = f"""
다음은 '{name}'라는 스타트업에 관한 정보입니다:

{summary}

위 정보를 바탕으로, 이 스타트업이 속한 도메인을 아래 목록 중에서 하나만 선택하여 답변해주세요:
헬스케어, AI, 핀테크, 에듀테크, 푸드테크, 커머스, 교육, 모빌리티, 에너지, 환경, 소프트웨어, 하드웨어, 소셜미디어, 엔터테인먼트, 부동산, 보안

한 단어로만 대답해주세요. 위 목록에 없다면 가장 근접한 것을 선택하거나 '기타'라고 답변하세요.
"""
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 스타트업 분석가입니다. 주어진 정보를 바탕으로 스타트업의 도메인을 분류하세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.3
        )
        
        domain = response.choices[0].message.content.strip()
        logger.info(f"스타트업 '{name}'의 도메인으로 '{domain}'을(를) 추출했습니다.")
        return domain
        
    except Exception as e:
        logger.error(f"도메인 추출 중 오류 발생: {str(e)}")
        return "기술"  # 기본 도메인

# 랭그래프 노드 생성 함수 - 메인에서 임포트할 때 사용
def create_startup_exploration_agent():
    from langgraph.graph import StateGraph
    
    graph = StateGraph(InvestmentState)
    graph.add_node("startup_exploration", startup_exploration)
    
    # 시작점과 종료점이 같은 단일 노드 그래프
    graph.set_entry_point("startup_exploration")
    graph.set_finish_point("startup_exploration")
    
    return graph.compile()

# 테스트 코드
if __name__ == "__main__":
    # 환경 변수에서 API 키 가져오기
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        logger.error("환경 변수에 API 키가 설정되지 않았습니다.")
        exit(1)
    
    # 리소스 초기화
    init_resources(
        pinecone_api_key=PINECONE_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        index_name="startup-index"  # Pinecone 인덱스 이름
    )
    
    # 테스트용 초기 상태
    test_state = {
        "user_query": "AI 기반 의료 진단 기술 스타트업",
        "startup_info": {},
        "competitors": [],
        "market_analysis": {},
        "investment_recommendation": {},
        "report_data": {},
        "status": "starting"
    }
    
    # 에이전트 실행
    result_state = startup_exploration(test_state)
    
    # 결과 출력
    print("\n=== 스타트업 탐색 결과 ===")
    print(f"스타트업 이름: {result_state['startup_info']['name']}")
    print(f"도메인: {result_state['startup_info']['domain']}")
    print(f"요약: {result_state['startup_info']['summary']}")