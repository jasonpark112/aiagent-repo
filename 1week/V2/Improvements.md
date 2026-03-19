# V2 Improvements

이 문서는 `V2`에서 시도한 개선점과 각 실험 결과를 누적 기록하기 위한 문서이다.
이후 개선 실험도 같은 형식으로 계속 추가한다.

## 1. Few-shot Prompting 추가

### 개선 목적

- 프롬프트에 예시 데이터를 넣어 모델이 분류 패턴과 JSON 응답 형식을 더 안정적으로 따르도록 유도한다.
- 특히 `urgency`와 `needs_clarification` 같은 경계 판단이 개선되는지 확인한다.

### 적용 내용

- 수정 파일: `1week/V2/prompts/inquiry_prompt.py`
- 변경 사항: 프롬프트 하단에 few-shot 예시 3건 추가

#### 추가한 예시

- 예시 1: 배송지 변경 문의 -> `order_change`, `medium`, `false`, `order_ops`
- 예시 2: 중복 결제 문의 -> `payment_issue`, `high`, `false`, `billing_ops`
- 예시 3: 단순 인사 -> `other`, `low`, `true`, `human_support`

### 평가 기준

- 정답 데이터: `1week/dataset.pretty.json`
- 실험 결과 데이터: `1week/V2/json/result/20260319-202059/analysis-results.json`
- 비교 기준: `1week/V1/V1PromptResult.md`

### 결과 요약

- 완전 일치(4개 필드 모두 일치): `9/12` = `75.0%`
- 필드 단위 정확도: `45/48` = `93.8%`

#### 필드별 정확도

- `intent`: `12/12` = `100%`
- `urgency`: `11/12` = `91.7%`
- `needs_clarification`: `10/12` = `83.3%`
- `route_to`: `12/12` = `100%`

### 토큰 사용량 분석

- 총 prompt tokens: `9,443`
- 평균 prompt tokens: `786.92`
- 총 completion tokens: `390`
- 평균 completion tokens: `32.5`
- 총 tokens: `9,833`

### 핵심 지표 분석

- 정답 기준 `needs_clarification=true` 필요 건수: `4건`
- 예측 `needs_clarification=true`: `2건`
- 정답 기준 `intent=other` 필요 건수: `3건`
- 예측 `intent=other`: `3건`
- 정답 기준 `route_to=human_support` 필요 건수: `3건`
- 예측 `route_to=human_support`: `3건`

### V1 결과와 비교

- 완전 일치율: `75.0% -> 75.0%`로 동일
- 필드 정확도: `91.7% -> 93.8%`로 `+2.1%p` 개선
- `urgency`: `75.0% -> 91.7%`로 개선
- `needs_clarification`: `91.7% -> 83.3%`로 하락
- `intent`, `route_to`: 모두 `100%` 유지
- 평균 prompt tokens: `612.92 -> 786.92`로 증가
- 총 tokens: `9,438 -> 9,833`으로 증가

### 세부 분석

#### 개선된 항목

- `ticket-08`: `urgency=low` -> `urgency=medium`으로 개선
- `ticket-12`: `urgency=low` -> `urgency=medium`으로 개선

#### 여전히 오답인 항목

- `ticket-09`: 기대값 `urgency=high`, 예측값 `urgency=medium`
- `ticket-12`: 기대값 `needs_clarification=true`, 예측값 `needs_clarification=false`

#### 새롭게 악화된 항목

- `ticket-11`: 기대값 `needs_clarification=true`, 예측값 `needs_clarification=false`

### 해석

- few-shot 예시 추가는 `urgency` 판단 보정에는 효과가 있었다.
- 반면 모델이 애매한 상담형 문의를 더 쉽게 확정적으로 해석하면서 `needs_clarification` 성능은 하락했다.
- `intent=other`, `route_to=human_support`의 예측 건수는 줄지 않았다.
- 즉, 현재 few-shot은 "긴급도 판단 강화"에는 도움을 주었지만, "애매함 인식" 개선이나 보수적 라우팅 감소에는 도움이 없었다.

### 결론

- 정확도는 소폭 개선되었지만, 토큰 사용량이 늘었고 핵심 목표였던 `other/human_support` 감소 효과는 없었다.

### 다음 개선 방향

- `needs_clarification=true`가 되어야 하는 경계 사례 예시를 추가한다.
- `ticket-11`, `ticket-12`와 유사한 절차 상담형 문의를 예시에 포함한다.
- "정보 문의/절차 상담"과 "즉시 처리 요청"의 구분 기준을 프롬프트 규칙에 더 명확히 적는다.

## 2. 불필요한 프롬프트 축소

### 개선 목적

- `pydantic`의 `json_schema`가 형식과 허용값을 강제하므로, 프롬프트에서 중복되는 설명과 예시를 줄여 prompt token 사용량을 낮춘다.
- 토큰 절감이 실제 정확도와 핵심 지표에 어떤 영향을 주는지 확인한다.

### 적용 내용

- 수정 파일: `1week/V2/prompts/inquiry_prompt.py`
- 변경 사항:
- 출력 형식 강제 문구와 few-shot 예시 제거
- 중복 설명을 줄이고 핵심 분류 규칙만 남긴 압축 프롬프트로 변경

### 평가 기준

- 정답 데이터: `1week/dataset.pretty.json`
- 실험 결과 데이터: `1week/V2/json/result/20260319-203207/analysis-results.json`
- 비교 기준:
- 베이스라인 `1week/V2/V1PromptResult.md`
- 1차 개선 `1week/V2/json/result/20260319-202059`

### 결과 요약

- 완전 일치(4개 필드 모두 일치): `8/12` = `66.7%`
- 필드 단위 정확도: `43/48` = `89.6%`

#### 필드별 정확도

- `intent`: `12/12` = `100%`
- `urgency`: `8/12` = `66.7%`
- `needs_clarification`: `11/12` = `91.7%`
- `route_to`: `12/12` = `100%`

### 토큰 사용량 분석

- 총 prompt tokens: `5,147`
- 평균 prompt tokens: `428.92`
- 총 completion tokens: `358`
- 평균 completion tokens: `29.83`
- 총 tokens: `7,829`

### 핵심 지표 분석

- 정답 기준 `needs_clarification=true` 필요 건수: `4건`
- 예측 `needs_clarification=true`: `5건`
- 정답 기준 `intent=other` 필요 건수: `3건`
- 예측 `intent=other`: `3건`
- 정답 기준 `route_to=human_support` 필요 건수: `3건`
- 예측 `route_to=human_support`: `3건`

### 이전 결과와 비교

- 베이스라인 대비 완전 일치율: `75.0% -> 66.7%`로 하락
- 베이스라인 대비 필드 정확도: `93.8% -> 89.6%`로 하락
- 1차 개선 대비 완전 일치율: `75.0% -> 66.7%`로 하락
- 1차 개선 대비 필드 정확도: `93.8% -> 89.6%`로 하락
- 베이스라인 대비 평균 prompt tokens: `612.92 -> 428.92`로 감소
- 1차 개선 대비 평균 prompt tokens: `786.92 -> 428.92`로 크게 감소
- 베이스라인 대비 총 tokens: `9,438 -> 7,829`로 감소
- 1차 개선 대비 총 tokens: `9,833 -> 7,829`로 감소

### 세부 분석

#### 악화된 항목

- `ticket-08`: 기대값 `urgency=medium`, `needs_clarification=false` / 예측값 `urgency=low`, `needs_clarification=true`
- `ticket-09`: 기대값 `urgency=high` / 예측값 `urgency=medium`
- `ticket-11`: 기대값 `urgency=medium` / 예측값 `urgency=low`
- `ticket-12`: 기대값 `urgency=medium` / 예측값 `urgency=low`

#### 유지된 항목

- `intent=other` 예측 건수는 `3건`으로 그대로였다.
- `route_to=human_support` 예측 건수도 `3건`으로 그대로였다.

### 해석

- 프롬프트 축소는 prompt token 절감에는 확실히 효과가 있었다.
- 그러나 `urgency` 판단이 전반적으로 보수적으로 낮아졌고, `needs_clarification=true`도 필요 이상으로 더 많이 출력되었다.
- 특히 중요하게 본 `intent=other`, `route_to=human_support` 예측 건수는 전혀 줄지 않았다.
- 즉, 이번 실험은 비용 절감은 성공했지만 분류 품질과 핵심 목표 달성에는 실패한 실험이다.

### 결론

- 토큰 절감 효과 자체는 유의미합니다.
- 하지만 정확도와 핵심 운영 지표가 더 나빠졌기 때문에 실험 결과는 실패에 가깝습니다.
- 특히 목표가 other/human_support 억제였다면 2번째 개선은 그 목표에도 도움이 안 됐습니다.

### 다음 개선 방향

- 프롬프트를 다시 길게 늘리기보다, 경계 사례를 잡는 짧은 규칙만 일부 복원한다.
- `ticket-08` 같은 환불 가능 여부 문의는 `refund_exchange`로 보되 `needs_clarification`을 과도하게 켜지 않도록 기준을 분명히 적는다.
- `ticket-11`은 `other/human_support`로 가더라도 `urgency`를 `low`로 내리지 않도록 보완 규칙을 추가한다.
- `ticket-12` 같은 절차 상담형 문의는 `needs_clarification=true`를 유지하되 `urgency`는 `medium`으로 판단하도록 규칙을 보강한다.
