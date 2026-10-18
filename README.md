# 한강 유람선 (han-cruise)

서울 한강 유람선 갑판에서 여의도 → 원효대교 → 한강철교 → 노들섬 앞 회항 구간을 관람하는 실시간 WebGPU 앱입니다. (작업 진행 중)

```
npm install
npm run dev        # http://127.0.0.1:5173/
```

- 데이터 재생성: `npm run fetch-data && npm run build-scene`
- 항로 검증: `node tools/check-route.mjs`
- 검증 장면 캡처: `node tools/capture.mjs` / 성능: `node tools/profile.mjs`

라이선스·출처: `THIRD_PARTY_NOTICES.md`
