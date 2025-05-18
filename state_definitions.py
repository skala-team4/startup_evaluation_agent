from typing import List, Optional
from pydantic import BaseModel


# 📌 후보 스타트업 문서
class CandidateDocument(BaseModel):
    user_query: str
    name: str
    summary: str
    domain: str


# 📌 시장성 점수
class MarketScores(BaseModel):
    market_size: int                       # 시장 크기
    problem_fit: int                       # 문제-제품 적합성
    willingness_to_pay: int               # 고객의 지불 의사
    revenue_model_clarity: int            # 수익 모델 명확성
    upside_potential: int                 # 성장 가능성


# 📌 경쟁사 문서 요약
class CompetitorDocument(BaseModel):
    title: str
    url: str
    content: str


# 📌 경쟁사 분석 결과
class MarketAnalysis(BaseModel):
    competitor_documents: Optional[List[CompetitorDocument]] = []
    competitive_score: Optional[float] = None
    competitive_reasoning: Optional[str] = None


# 📌 투자 판단 결과
class InvestmentDecision(BaseModel):
    judgement: Optional[str] = None       # 예: "통과", "불통과"
    reasoning: Optional[str] = None       # 판단 근거


# 📌 전체 LangGraph 상태
class GraphState(BaseModel):
    user_query: str                                           # 사용자 입력
    domain: Optional[str] = None                              # 추출된 산업 분야
    candidates_documents: Optional[List[CandidateDocument]] = []  # 스타트업 후보 목록
    evaluation_summary: Optional[str] = None                  # 후보 요약
    market_analysis: Optional[MarketAnalysis] = None          # 경쟁사 분석 결과
    market_scores: Optional[MarketScores] = None              # ✅ 시장성 점수 추가됨
    investment_decision: Optional[InvestmentDecision] = None  # 투자 여부
    final_report: Optional[str] = None                        # 최종 보고서