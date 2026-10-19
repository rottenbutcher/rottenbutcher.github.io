/* ==========================================================
   상단 메뉴 / 푸터 — 여기만 고치면 모든 페이지에 반영됩니다.
   ========================================================== */
const NAV = [
  { key: "about",        href: "index.html#about",  label: "About" },
  { key: "research",     href: "research.html",     label: "Research &amp; Projects" },
  { key: "publications", href: "publications.html", label: "Presentations &amp; Publications" },
  { key: "awards",       href: "awards.html",       label: "Awards &amp; Honors" },
  { key: "hobby",        href: "hobby.html",        label: "Hobby" },
];
const CV_FILE = "assets/cv.pdf";
const CV_NAME = "Junseo_Lee_CV.pdf";
const LINKS = {
  email: "mailto:junseo777@snu.ac.kr",
  github: "https://github.com/rottenbutcher",
  linkedin: "https://www.linkedin.com/in/jslee1211",
};

(function renderChrome() {
  const page = document.body.dataset.page || "";
  const navItems = NAV.map((n) => {
    // On the home page, "About" scrolls in place
    const href = page === "home" && n.key === "about" ? "#about" : n.href;
    const active = n.key === page ? ' class="active" aria-current="page"' : "";
    return `<li><a href="${href}"${active}>${n.label}</a></li>`;
  }).join("");

  const header = document.querySelector(".site-header");
  if (header) {
    header.innerHTML = `
      <div class="wrap">
        <a class="brand" href="${page === "home" ? "#top" : "index.html"}">JUNSEO LEE<span>.</span></a>
        <ul class="nav">${navItems}</ul>
        <div class="header-right">
          <a class="pill" href="${CV_FILE}" download="${CV_NAME}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>
            CV
          </a>
          <button class="menu-btn" aria-label="Menu" aria-expanded="false">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
          </button>
        </div>
      </div>`;
  }

  const footer = document.querySelector(".site-footer");
  if (footer) {
    footer.innerHTML = `
      <div class="wrap">
        <span>© ${new Date().getFullYear()} Junseo Lee</span>
        <span class="links">
          <a href="${LINKS.email}">Email</a>
          <a href="${LINKS.github}" target="_blank" rel="noopener">GitHub</a>
          <a href="${LINKS.linkedin}" target="_blank" rel="noopener">LinkedIn</a>
        </span>
      </div>`;
  }
})();

(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Background video ---------- */
  const video = document.querySelector(".bg-video");
  if (video) {
    if (reduceMotion) {
      video.removeAttribute("autoplay");
      video.pause();
    } else {
      // Mobile gets the lighter file
      const small = window.matchMedia("(max-width: 760px)").matches;
      const mp4 = video.canPlayType('video/mp4; codecs="avc1.640028"');
      const ext = mp4 ? "mp4" : "webm";
      const base = small ? "assets/bg-mobile." : "assets/bg.";
      video.src = base + ext;
      video.play().catch(() => {});
    }
  }

  /* ---------- Header + background dim on scroll ---------- */
  const header = document.querySelector(".site-header");
  const root = document.documentElement;
  const onScroll = () => {
    const y = window.scrollY;
    const vh = window.innerHeight;
    header.classList.toggle("scrolled", y > 40);
    // 0.15 at the top → 0.75 after one screen of scrolling
    const dim = 0.15 + Math.min(y / vh, 1) * 0.6;
    root.style.setProperty("--dim", dim.toFixed(3));
  };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  /* ---------- Mobile menu ---------- */
  const btn = document.querySelector(".menu-btn");
  const nav = document.querySelector(".nav");
  const closeMenu = () => {
    nav.classList.remove("open");
    header.classList.remove("menu-open");
    btn.setAttribute("aria-expanded", "false");
  };
  btn.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    header.classList.toggle("menu-open", open);
    btn.setAttribute("aria-expanded", String(open));
  });
  nav.querySelectorAll("a").forEach((a) => a.addEventListener("click", closeMenu));

  /* ---------- Active nav link ---------- */
  const links = [...nav.querySelectorAll('a[href^="#"]')];
  const sections = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  if ("IntersectionObserver" in window) {
    const spy = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          links.forEach((a) => {
            if (a.getAttribute("href") === "#" + e.target.id) a.classList.toggle("active", e.isIntersecting);
          });
        });
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    sections.forEach((s) => spy.observe(s));

    /* ---------- Reveal on scroll ---------- */
    if (!reduceMotion) {
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((e) => {
            if (e.isIntersecting) {
              e.target.classList.add("in");
              io.unobserve(e.target);
            }
          });
        },
        { rootMargin: "0px 0px -10% 0px" }
      );
      document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
    } else {
      document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in"));
    }
  } else {
    document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in"));
  }

})();
