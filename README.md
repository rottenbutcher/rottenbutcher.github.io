# rottenbutcher.github.io

Junseo Lee의 개인 홈페이지 (GitHub Pages, 순수 HTML/CSS).

## 폴더 구조

```
index.html          홈 (상단 배너 + 프로필 + 바로가기)
research.html       Research & Projects
publications.html   Presentations & Publications
awards.html         Awards
hobby.html          Hobby
.nojekyll           GitHub Pages가 파일을 그대로 서빙하도록
assets/
  style.css         전체 디자인
  main.js           상단 메뉴(배너) + 푸터 — 메뉴 수정은 여기서 한 번만
  favicon.svg
  banner.jpg        ← 직접 넣기: 상단 배너 사진 (가로 2000px 정도 권장)
  profile.jpg       ← 직접 넣기: 프로필 사진 (세로 4:5 비율 권장)
  cv.pdf            ← 직접 넣기: CV PDF (상단 CV 버튼 클릭 시 자동 다운로드)
```

사진·CV 파일이 아직 없으면 배너는 남색 그라데이션, 프로필은 "JL" 이니셜로 표시됩니다.
파일 이름만 위와 똑같이 맞춰 넣으면 자동으로 바뀝니다.

## 배포 (GitHub Pages)

- 저장소 이름: `rottenbutcher.github.io` → 사이트 주소 https://rottenbutcher.github.io
- Settings → Pages: Source `Deploy from a branch`, Branch `main` / `(root)`
- `main`에 푸시하면 1~2분 뒤 자동 반영됩니다.

나중에 개인 도메인을 쓰려면 루트에 `CNAME` 파일(내용: 도메인 이름 한 줄)을 추가하고 DNS를 GitHub Pages로 연결하면 됩니다.

## 내용 수정

각 페이지의 `<li class="entry"> … </li>` 블록이 항목 하나입니다. 복사해서 늘리고,
`<!-- … -->` 주석으로 표시해 둔 자리(기간, 설명, 발표 제목, 공동저자 등)를 채우면 됩니다.
