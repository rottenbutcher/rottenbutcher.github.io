<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,50:203A43,100:2C5364&height=200&section=header&text=rottenbutcher.github.io&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=Junseo%20Lee%20%C2%B7%20Personal%20Homepage&descAlignY=60&descSize=18&animation=fadeIn" width="100%" alt="rottenbutcher.github.io"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=20&pause=1200&color=2C9CDB&center=true&vCenter=true&width=600&lines=Research+%C2%B7+Publications+%C2%B7+Awards+%C2%B7+Hobby;Pure+HTML+%2B+CSS+%E2%80%94+no+build+step;Hosted+on+GitHub+Pages" alt="typing banner"/>

<br/>

**🏠 Junseo Lee의 개인 홈페이지 — GitHub Pages, 순수 HTML / CSS**

<br/>

[![Website](https://img.shields.io/badge/Website-rottenbutcher.github.io-2C5364?style=for-the-badge&logo=googlechrome&logoColor=white)](https://rottenbutcher.github.io)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deployed-222222?style=for-the-badge&logo=githubpages&logoColor=white)](https://pages.github.com/)

[![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/docs/Web/HTML)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)](https://developer.mozilla.org/docs/Web/CSS)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/docs/Web/JavaScript)
[![WebP](https://img.shields.io/badge/Images-WebP-4285F4?logo=google&logoColor=white)](https://developers.google.com/speed/webp)

<br/>

[📄 Pages](#-pages) •
[🗂️ 폴더 구조](#️-폴더-구조) •
[🎬 배경 영상](#-배경-영상-교체) •
[🚀 배포](#-배포-github-pages) •
[✏️ 내용 수정](#️-내용-수정)

</div>

---

## 📄 Pages

| | Page | 파일 | 내용 |
|:-:|---|---|---|
| 🏠 | **Home** | `index.html` | 배경 영상 + About + Recent |
| 🔬 | **Research** | `research.html` | Research & Projects |
| 📚 | **Publications** | `publications.html` | Presentations & Publications |
| 🏆 | **Awards** | `awards.html` | Awards & Honors |
| 🎨 | **Hobby** | `hobby.html` | Hobby |

---

## 🗂️ 폴더 구조

```text
.
├── index.html            🏠 메인 (배경 영상 + About + Recent)
├── research.html         🔬 Research & Projects
├── publications.html     📚 Presentations & Publications
├── awards.html           🏆 Awards & Honors
├── hobby.html            🎨 Hobby
├── .nojekyll             ⚙️ GitHub Pages가 파일을 그대로 서빙하도록
└── assets/
    ├── site.css          🎨 전체 디자인 (강조색은 맨 위 --accent 한 줄로 변경)
    ├── site.js           ⚡ 배경 영상 재생, 스크롤 효과, 모바일 메뉴
    ├── bg.mp4 / bg.webm                  🎬 배경 영상 (PC)
    ├── bg-mobile.mp4 / bg-mobile.webm    📱 배경 영상 (모바일, 가벼운 버전)
    ├── bg-poster.jpg     🖼️ 영상 로드 전 / '동작 줄이기' 설정 시 정지 화면
    ├── img/              📷 사진·로고 (WebP)
    │   └── profile.webp  👤 프로필 사진 (배경 제거)
    ├── apple-touch-icon.png  📲 휴대폰 홈 화면 아이콘
    ├── cv.pdf            📄 CV (상단 CV 버튼 클릭 시 다운로드)
    └── favicon.svg       🔖 파비콘
```

> [!NOTE]
> 📷 **이미지 규칙**
> - 사진: 목록용 `*-sm.webp` + 크게 보기용 `*.webp`
> - 로고: 흰 배경을 지우고 흰색 단색으로 변환한 `logo-*.webp`
>
> 🧭 상단 메뉴는 `site.js`가 아니라 **각 HTML 파일 안**에 있습니다.

> [!TIP]
> 🎨 사이트 강조색은 `assets/site.css` 맨 위의 `--accent` **한 줄**만 바꾸면 전체에 반영됩니다.

---

## 🎬 배경 영상 교체

같은 이름으로 파일을 덮어쓰면 됩니다.

| 파일 | 용도 |
|---|---|
| `bg.mp4` (H.264) | 💻 PC — 대부분의 브라우저 |
| `bg.webm` | 💻 PC — 그 외 브라우저 |
| `bg-mobile.mp4` / `bg-mobile.webm` | 📱 모바일용 가벼운 버전 |
| `bg-poster.jpg` | 🖼️ 로딩 전 / 동작 줄이기 설정 시 정지 화면 |

---

## 🚀 배포 (GitHub Pages)

| 항목 | 설정 |
|---|---|
| 📦 저장소 이름 | `rottenbutcher.github.io` |
| 🌐 사이트 주소 | <https://rottenbutcher.github.io> |
| ⚙️ Settings → Pages | Source: `Deploy from a branch` · Branch: `main` / `(root)` |
| ⏱️ 반영 시간 | `main`에 push 후 **1~2분** 뒤 자동 반영 |

<details>
<summary>🌍 <b>개인 도메인 연결하기</b></summary>

<br/>

1. 저장소 루트에 `CNAME` 파일을 만들고, 내용으로 도메인 이름 **한 줄**을 적습니다.
2. 도메인의 DNS를 GitHub Pages로 연결합니다.

</details>

---

## ✏️ 내용 수정

각 페이지의 `<li class="entry"> … </li>` 블록 하나가 **항목 하나**입니다.

1️⃣ 블록을 복사해서 항목을 늘리고<br/>
2️⃣ `<!-- … -->` 주석으로 표시해 둔 자리(기간, 설명, 발표 제목, 공동저자 등)를 채우면 끝! ✅

```html
<li class="entry">
  <!-- 기간 -->
  <!-- 제목 / 설명 -->
  <!-- 공동저자 등 -->
</li>
```

<div align="center">

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2C5364,50:203A43,100:0F2027&height=120&section=footer" width="100%" alt=""/>

</div>
