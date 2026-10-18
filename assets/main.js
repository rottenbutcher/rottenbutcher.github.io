/* Shared header (top banner) + footer for every page.
   메뉴를 바꾸고 싶으면 아래 NAV 배열만 수정하면 모든 페이지에 반영됩니다. */

const NAV = [
  { href: "research.html",     label: "Research & Projects" },
  { href: "publications.html", label: "Presentations & Publications" },
  { href: "awards.html",       label: "Awards" },
  { href: "hobby.html",        label: "Hobby" },
];

const CV_FILE = "assets/cv.pdf";              // 여기에 CV PDF를 넣으세요
const CV_DOWNLOAD_NAME = "Junseo_Lee_CV.pdf"; // 다운로드될 때 파일 이름

const ICON_DOWNLOAD = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>';
const ICON_MENU = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>';

(function renderChrome() {
  const here = location.pathname.split("/").pop() || "index.html";

  const links = NAV.map(
    (n) => `<li><a href="${n.href}"${n.href === here ? ' class="active" aria-current="page"' : ""}>${n.label}</a></li>`
  ).join("");

  const header = document.createElement("header");
  header.className = "site-nav";
  header.innerHTML = `
    <div class="container">
      <a class="brand" href="index.html">Junseo Lee<span>.</span></a>
      <button class="nav-toggle" aria-label="Open menu" aria-expanded="false">${ICON_MENU}</button>
      <ul class="nav-links">
        ${links}
        <li><a class="cv-btn" href="${CV_FILE}" download="${CV_DOWNLOAD_NAME}">${ICON_DOWNLOAD}CV</a></li>
      </ul>
    </div>`;
  document.body.prepend(header);

  const footer = document.createElement("footer");
  footer.className = "site-footer";
  footer.innerHTML = `
    <div class="container">
      <span>© ${new Date().getFullYear()} Junseo Lee</span>
      <span><a href="https://github.com/rottenbutcher" target="_blank" rel="noopener">GitHub</a></span>
    </div>`;
  document.body.append(footer);

  const toggle = header.querySelector(".nav-toggle");
  const menu = header.querySelector(".nav-links");
  toggle.addEventListener("click", () => {
    const open = menu.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
})();
