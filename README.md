# Curl Up & Dye — Dreamweaver-Friendly Redesign

This project is a static, editable website redesign for **Curl Up & Dye**, the independent San Francisco hairstylist brand by **Sam Strock**.

## File Structure (Dreamweaver-Friendly)

- `index.html` — Homepage (conversion-focused landing page)
- `services.html` — Service categories with starting prices
- `portfolio.html` — Image-first portfolio with JS filters + lightbox
- `about.html` — Brand story and stylist bio
- `reviews.html` — Testimonial page
- `faq.html` — FAQ and policy page
- `contact.html` — Contact details, hours, booking CTA
- `style-guide.html` — Visual system reference for future edits
- `css/styles.css` — Main stylesheet
- `js/main.js` — Lightweight interactions only
- `images/` — Local placeholder SVG images (replace with optimized photos)

## Optional PHP Include Structure (If Needed Later)

If you want reusable partials while keeping Dreamweaver compatibility:

- `/includes/header.php`
- `/includes/footer.php`
- `/includes/final-cta.php`

Then each page can switch from `.html` to `.php` and include partials with:

```php
<?php include 'includes/header.php'; ?>
```

## Conversion + UX Features Implemented

- Sticky header with prominent **Book Now** CTA
- Mobile menu with vanilla JS only
- Mobile sticky **Book Now** button
- Homepage with modular sections and repeated booking prompts
- Portfolio category filters and lightweight modal/lightbox
- FAQ with semantic `<details>` for accessibility and performance
- Scannable service cards and clear policy language

## Suggested Homepage Copy Blocks

1. **Hero headline:**
   - *Curly texture mastery. Dimensional blonding. Modern cuts that move.*
2. **Subhead:**
   - *I’m Sam Strock, a San Francisco hairstylist delivering tailored cuts and color with a one-on-one, inclusive approach.*
3. **Primary CTA:** Book Now
4. **Secondary CTA:** View Portfolio
5. **Final CTA:**
   - *Ready for hair that feels like you, only sharper?*

## SEO Metadata Pattern (Use Across Pages)

- **Title format:** `Page Topic | Curl Up & Dye San Francisco`
- **Meta descriptions:** 145–160 chars, include local terms naturally
- Use one clear `<h1>` per page
- Keep heading hierarchy in order (`h1 > h2 > h3`)
- Include image `alt` text that describes hair result, texture, tone, or cut
- Keep NAP visible on homepage/contact page
- Local schema example included on `index.html` (`HairSalon`)

## Sample Content Structure for Portfolio Entries

For each gallery image:

- Category tags (`curls`, `blondes`, `cuts`, `vivid`, `transformations`)
- Image alt text that names the visible result
- Optional short caption in future versions:
  - service type
  - maintenance cadence
  - tonal direction

## Sample Content Structure for Services

Each service card should include:

- Service name
- Plain-language description
- Starting price
- “Ideal for” guidance

Keep this short and mobile-scannable.

## Image Replacement Notes

Current images are lightweight SVG placeholders so the prototype renders immediately. Replace with optimized real photography:

- Export WebP/JPEG at responsive sizes
- Keep file names descriptive
- Aim for under 250 KB per non-hero image when possible

## Style Guide

Open `style-guide.html` to view:

- Brand color tokens
- Typography pairings
- Spacing rhythm
- Button styles
- Card patterns

