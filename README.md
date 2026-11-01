# Orville Rockwell Jr. — Academic Website

Live at **https://OrvilleRockwellJr.github.io**

A static site. No build step, no dependencies, no framework — plain HTML and CSS with one
small inline script. Every file in this repo is served as-is.

```
index.html                     the whole site (inline CSS + JS)
404.html                       not-found page
favicon.svg                    tab icon
robots.txt                     crawler rules
sitemap.xml                    search-engine sitemap
assets/img/profile.jpg         portrait
assets/img/camera-array.jpg    research card photo (Brown / B-CamSync)
assets/img/rov.jpg             NUWave ROV photo
assets/img/cv-preview.jpg      CV page rendered for the in-page viewer
assets/img/og.jpg              1200x630 link-preview card
assets/img/apple-touch-icon.png
cv/Orville_Rockwell_CV.pdf     CV, linked from the nav, hero and footer
```

## Updating

Edit, then:

```bash
git add . && git commit -m "update" && git push
```

GitHub Pages redeploys within a minute or so. Hard-refresh (Ctrl+Shift+R) if you still see
the old version — Pages caches aggressively.

## Editing the content

Everything is in `index.html`. Sections in order: hero, Research Interests, Research,
Projects, Get In Touch.

- **Add a research or project card** — copy an existing `<article class="card ...">` block
  and edit it. The second class sets the color: `card auto` (cyan, autonomous systems),
  `card ml` (indigo, machine learning), `card bio` (rose, computational biology). The grid
  reflows on its own.
- **Change the palette** — the `:root` block at the top of the `<style>`.
- **Swap the portrait** — replace `assets/img/profile.jpg`; square crops work best.
- **Update the CV** — replace `cv/Orville_Rockwell_CV.pdf`, and regenerate
  `assets/img/cv-preview.jpg` (a 150 dpi render of page 1) so the in-page viewer matches.

One caution: the CSS classes `auto`, `ml` and `bio` carry the thread colors. Don't reuse
those names for anything else or the styles will collide.

## Custom domain (optional)

The GitHub Student Developer Pack (education.github.com/pack) includes a free first-year
domain — Namecheap `.me`, Name.com, and `.TECH` are all listed. To use one:

1. Add a `CNAME` file containing only the domain, e.g. `orvillerockwell.me`
2. At the registrar, add four `A` records for `@` → `185.199.108.153`, `185.199.109.153`,
   `185.199.110.153`, `185.199.111.153`, and a `CNAME` for `www` → `OrvilleRockwellJr.github.io`
3. Settings → Pages → enter the domain → tick **Enforce HTTPS** once the cert issues
4. Update the URLs in `index.html` (canonical, Open Graph, Twitter, JSON-LD), `robots.txt`
   and `sitemap.xml` from `OrvilleRockwellJr.github.io` to the new domain

## Local preview

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000 — use a server rather than opening `index.html` directly,
so relative paths behave as they do in production.
