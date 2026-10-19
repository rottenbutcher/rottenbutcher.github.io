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
