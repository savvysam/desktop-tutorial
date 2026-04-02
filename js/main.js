// Curl Up & Dye main interactions (vanilla JS only)
(function () {
  const navToggle = document.querySelector('[data-nav-toggle]');
  const nav = document.querySelector('[data-nav]');

  if (navToggle && nav) {
    navToggle.addEventListener('click', function () {
      const isOpen = nav.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', String(isOpen));
    });

    nav.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        nav.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  const filterButtons = document.querySelectorAll('[data-filter]');
  const filterItems = document.querySelectorAll('[data-category]');

  filterButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      const category = button.getAttribute('data-filter');

      filterButtons.forEach(function (btn) {
        btn.classList.remove('is-active');
        btn.setAttribute('aria-pressed', 'false');
      });

      button.classList.add('is-active');
      button.setAttribute('aria-pressed', 'true');

      filterItems.forEach(function (item) {
        const itemCategory = item.getAttribute('data-category');
        const shouldShow = category === 'all' || itemCategory === category;
        item.hidden = !shouldShow;
      });
    });
  });

  const lightbox = document.querySelector('[data-lightbox]');
  const lightboxImage = document.querySelector('[data-lightbox-image]');
  const galleryTriggers = document.querySelectorAll('[data-lightbox-src]');

  if (lightbox && lightboxImage && galleryTriggers.length) {
    galleryTriggers.forEach(function (trigger) {
      trigger.addEventListener('click', function () {
        const src = trigger.getAttribute('data-lightbox-src');
        const alt = trigger.getAttribute('data-lightbox-alt') || 'Portfolio image';

        lightboxImage.setAttribute('src', src);
        lightboxImage.setAttribute('alt', alt);
        lightbox.classList.add('open');
      });
    });

    lightbox.addEventListener('click', function (event) {
      if (event.target === lightbox || event.target.hasAttribute('data-lightbox-close')) {
        lightbox.classList.remove('open');
      }
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') {
        lightbox.classList.remove('open');
      }
    });
  }
})();
