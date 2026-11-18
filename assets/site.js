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


  /* ---------- Lightbox for photos ---------- */
  const photos = document.querySelectorAll("a.photo");
  if (photos.length) {
    const box = document.createElement("figure");
    box.className = "lightbox";
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-modal", "true");
    box.innerHTML = '<button class="close" aria-label="Close">×</button><img alt=""><figcaption></figcaption>';
    document.body.appendChild(box);
    const img = box.querySelector("img");
    const cap = box.querySelector("figcaption");
    let last = null;
    const close = () => { box.classList.remove("open"); document.body.style.overflow = ""; if (last) last.focus(); };
    photos.forEach((a) => a.addEventListener("click", (e) => {
      e.preventDefault();
      last = a;
      img.src = a.getAttribute("href");
      img.alt = a.dataset.caption || "";
      cap.textContent = a.dataset.caption || "";
      box.classList.add("open");
      document.body.style.overflow = "hidden";
      box.querySelector(".close").focus();
    }));
    box.addEventListener("click", close);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && box.classList.contains("open")) close(); });
  }
})();

/* ---------- itch.io games & stats (Hobby page) ----------
   Data: assets/data/itch-stats.json, refreshed hourly by .github/workflows/itch-stats.yml.
   - Games already written in hobby.html get their count filled in.
   - Any other published game on itch.io gets a tile added automatically.
   Browser games show views; downloadable games show downloads. */
(function () {
  const grid = document.querySelector(".games");
  if (!grid) return;
  const esc = (t) => String(t == null ? "" : t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const statText = (g) => {
    const browser = g.type === "html";
    const n = browser ? g.views : g.downloads;
    return typeof n === "number" ? n.toLocaleString("en-US") + (browser ? " views" : " downloads") : "";
  };
  fetch("assets/data/itch-stats.json?t=" + Date.now(), { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((data) => {
      if (!data || !data.games) return;
      const known = new Set();
      document.querySelectorAll("[data-itch-url]").forEach((el) => {
        const url = el.dataset.itchUrl.replace(/\/$/, "");
        known.add(url);
        const g = data.games[url];
        if (!g) return;
        const t = statText(g);
        if (t) { el.textContent = t; el.hidden = false; }
      });
      Object.entries(data.games).forEach(([url, g]) => {
        if (known.has(url) || g.published === false || !/^https:\/\//.test(url)) return;
        const a = document.createElement("a");
        a.className = "game";
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener";
        const coverSrc = g.cover_local || (g.cover && /^https:\/\//.test(g.cover) ? g.cover : "");
        const cover = coverSrc
          ? `<img src="${esc(coverSrc)}" alt="${esc(g.title)} cover" width="315" height="250" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'game-nocover',textContent:this.alt.replace(/ cover$/,'')}))">`
          : `<span class="game-nocover">${esc(g.title)}</span>`;
        const t = statText(g);
        a.innerHTML = `${cover}
          <span class="game-body">
            <strong>${esc(g.title)}</strong>
            <span class="game-meta">${g.genre ? esc(g.genre.replace(/, /g, " · ")) + " · " : ""}${g.type === "html" ? "Play in browser" : "Download"}</span>
            ${t ? `<span class="game-stats">${esc(t)}</span>` : ""}
            ${(g.short_text || g.description) ? `<span class="game-desc">${esc(g.short_text || g.description)}</span>` : ""}
          </span>`;
        grid.appendChild(a);
      });
    })
    .catch(() => {});
})();
