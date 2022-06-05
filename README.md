# Tri-Former-2022-1_KHU_CD1
## KHU 2022-1학기 캡스톤 디자인1 [2pm] Tri-Former

#### Mixed Triformer

>* 'tri_decomp' : 사용 trend, seasonal, noise 분리
>* 'series_decomp' : encoder, decoder에는 기존 decomp 사용
>* 'auto-correlation block' : 활용안함 (기존 autoformer처럼 seasonal에는 적용)
>* noise는 최종 prediction에서 제거

#### No autocorrelation ver

>* 'tri_decomp' : 사용 trend, seasonal, noise 분리
>* 'auto-correlation block' : 활용안함 (기존 autoformer처럼 seasonal에는 적용)

#### with autocorrelation ver

>* 'tri_decomp' : 사용 trend, seasonal, noise 분리
>* 'auto-correlation block' : trend에만 적용 (기존 autoformer처럼 seasonal에는 적용)

#### full residual ver

>* 'tri_decomp' : 사용 trend, seasonal, noise 분리
>* 'auto-correlation block' : 활용안함 (기존 autoformer처럼 seasonal에는 적용)
>* noise는 디코더 통과 없이 정답값으로 학습