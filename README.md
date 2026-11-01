# Orville Rockwell Jr. — Academic Website

Live at **https://orvillerockwelljr.com**

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

Everything is in `index.html`. Sections in order: hero, Research focus, News, Research,
Projects, Contact.

- **Add a news item** — copy an `<li>` in the News list; keep the newest first and update the
  "Last updated" line in the footer.
- **Add a research or project entry** — copy an existing `<article class="entry">` block.
  The colored dot before the title marks the thread: `var(--cv)` (cobalt, method),
  `var(--auto)` (teal, autonomous systems), `var(--bio)` (rose, computational biology).
- **Status tags** — `status live` (teal, ongoing), `status result` (rose, has a result),
  plain `status` (grey, in progress).
- **Change the palette** — the `:root` block at the top of the `<style>`.
- **Swap the portrait** — replace `assets/img/profile.jpg`; square crops work best.
- **Update the CV** — replace `cv/Orville_Rockwell_CV.pdf`, and regenerate
  `assets/img/cv-preview.jpg` (a 150 dpi render of page 1) so the in-page viewer matches.

The calibration-board figure on the Brown entry animates once when it scrolls into view
(skipped under reduced motion).

## Custom domain (optional)

The GitHub Student Developer Pack (education.github.com/pack) includes a free first-year
domain — Namecheap `.me`, Name.com, and `.TECH` are all listed. To use one:

1. Add a `CNAME` file containing only the domain, e.g. `orvillerockwell.me`
2. At the registrar, add four `A` records for `@` → `185.199.108.153`, `185.199.109.153`,
   `185.199.110.153`, `185.199.111.153`, and a `CNAME` for `www` → `orvillerockwelljr.com`
3. Settings → Pages → enter the domain → tick **Enforce HTTPS** once the cert issues
4. Update the URLs in `index.html` (canonical, Open Graph, Twitter, JSON-LD), `robots.txt`
   and `sitemap.xml` from `orvillerockwelljr.com` to the new domain

## Local preview

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000 — use a server rather than opening `index.html` directly,
so relative paths behave as they do in production.
