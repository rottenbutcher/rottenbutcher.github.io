# rottenbutcher.github.io

Junseo Lee의 개인 홈페이지 (GitHub Pages, 순수 HTML/CSS).

## 폴더 구조

```
index.html          메인 (배경 영상 + About + Recent)
research.html       Research & Projects
publications.html   Presentations & Publications
awards.html         Awards & Honors
hobby.html          Hobby
.nojekyll           GitHub Pages가 파일을 그대로 서빙하도록
assets/
  site.css          전체 디자인 (강조색은 맨 위 --accent 한 줄로 변경)
  site.js           상단 메뉴·푸터(NAV), 배경 영상 재생, 스크롤 효과, 모바일 메뉴
  bg.mp4 / bg.webm                 배경 영상 (PC)
  bg-mobile.mp4 / bg-mobile.webm   배경 영상 (모바일, 가벼운 버전)
  bg-poster.jpg     영상이 로드되기 전 / 동작 줄이기 설정 시 보이는 정지 화면
  profile.jpg       프로필 사진
  cv.pdf            CV (상단 CV 버튼 클릭 시 다운로드)
  favicon.svg
```

배경 영상을 바꾸려면 같은 이름으로 파일을 교체하세요. mp4(H.264)는 대부분의 브라우저용, webm은 그 외 브라우저용입니다.

## 배포 (GitHub Pages)

- 저장소 이름: `rottenbutcher.github.io` → 사이트 주소 https://rottenbutcher.github.io
- Settings → Pages: Source `Deploy from a branch`, Branch `main` / `(root)`
- `main`에 푸시하면 1~2분 뒤 자동 반영됩니다.

나중에 개인 도메인을 쓰려면 루트에 `CNAME` 파일(내용: 도메인 이름 한 줄)을 추가하고 DNS를 GitHub Pages로 연결하면 됩니다.

## 내용 수정

각 페이지의 `<li class="entry"> … </li>` 블록이 항목 하나입니다. 복사해서 늘리고,
`<!-- … -->` 주석으로 표시해 둔 자리(기간, 설명, 발표 제목, 공동저자 등)를 채우면 됩니다.
