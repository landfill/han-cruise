// Demonstration route (verified against OSM geometry, see docs/VERIFICATION.md):
// Yeouido E-Land Cruise pier -> under Wonhyo Bridge -> under the Hangang Railway Bridge ->
// turn in front of Nodeul island -> back under both bridges -> Yeouido pier.
// This is a route built for this app; it is not a published timetable of any operator.
// Coordinates: local metres (x east, z south) from tools/build-scene.mjs.
export const ROUTE_POINTS = [
  [-561, -106],  // berth on the outer face of the Yeouido E-Land Cruise pier, bow to the south-east
  [-515, -62],
  [-455, -95],
  [-370, -190],
  [-250, -290],
  [-110, -345],
  [8, -195],     // under Wonhyo Bridge, centre of the span between piers 7 and 8 (perpendicular)
  [100, -60],
  [330, 215],
  [540, 485],
  [653, 583],    // under the Hangang Railway Bridge (A line station 550 m, mid-span)
  [740, 645],
  [795, 664],
  [838, 640],    // turn to port in front of Nodeul island's west tip
  [836, 585],
  [790, 530],
  [736, 488],    // under the railway bridge again (A line station 676 m)
  [618, 394],
  [400, 160],
  [187, -110],
  [92, -251],    // under Wonhyo Bridge again (station 1102 m)
  [9, -376],
  [-200, -392],
  [-450, -360],
  [-620, -300],
  [-662, -228],
  [-620, -160],
  [-561, -106],  // alongside the pier again
];

export const LANDMARKS = [
  { key: '63', name: '63빌딩', pos: [-419, 569], height: 250, text: '1985년 준공. 지상 60층·높이 약 249m의 금색 유리 외벽 빌딩으로, 여의도 동쪽 끝에 서 있습니다.', range: 3500 },
  { key: 'wonhyo', name: '원효대교', pos: [50, -222], height: 20, text: '1981년 개통. 디비닥(FCM) 공법으로 지은 프리스트레스트 콘크리트 박스거더교로, 길이는 1,470m입니다.', range: 1600 },
  { key: 'rail', name: '한강철교', pos: [650, 576], height: 20, text: '1900년 A선 개통 — 한강에 처음 놓인 다리입니다. 지금은 A·B·C·D 네 개의 철도교가 나란히 놓여 있습니다.', range: 1500 },
  { key: 'hangang', name: '한강대교 · 노들섬', pos: [1192, 829], height: 20, text: '1917년 한강인도교로 처음 놓였습니다. 다리가 지나는 노들섬은 한강 가운데의 하중도입니다.', range: 1400 },
  { key: 'parc1', name: '파크원 타워', pos: [-1572, -186], height: 330, text: '2020년 준공. 높이 333m(69층)로 여의도에서 가장 높은 건물이며, 붉은 외부 골조가 특징입니다.', range: 4500 },
  { key: 'ifc', name: 'IFC 서울', pos: [-1698, -27], height: 280, text: '2012년 완공된 오피스 3개 동과 호텔로 이루어진 복합단지입니다. 가장 높은 Three IFC는 55층입니다.', range: 4500 },
  { key: 'namsan', name: 'N서울타워', pos: [3816, -2894], height: 490, text: '해발 약 262m 남산 정상에 선 높이 236.7m의 전파탑입니다. 1975년에 세워졌습니다.', range: 9000 },
  { key: 'mapo', name: '마포대교', pos: [-800, -914], height: 15, text: '1970년 개통. 여의도와 마포를 잇는 다리입니다.', range: 1400 },
];
