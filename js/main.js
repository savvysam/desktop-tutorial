// Curl Up & Dye interactions: mobile nav, portfolio filters, and lightbox.
(function () {
  var navToggle = document.querySelector('[data-nav-toggle]');
  var nav = document.querySelector('[data-nav]');

  if (navToggle && nav) {
    navToggle.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });

    nav.addEventListener('click', function (event) {
      if (event.target.matches('a')) {
        nav.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  var filterButtons = document.querySelectorAll('[data-filter]');
  var portfolioItems = document.querySelectorAll('[data-category]');

  if (filterButtons.length && portfolioItems.length) {
    filterButtons.forEach(function (button) {
      button.addEventListener('click', function () {
        var filter = button.getAttribute('data-filter');

        filterButtons.forEach(function (btn) {
          btn.classList.remove('active');
          btn.setAttribute('aria-pressed', 'false');
        });

        button.classList.add('active');
        button.setAttribute('aria-pressed', 'true');

        portfolioItems.forEach(function (item) {
          var category = item.getAttribute('data-category');
          var show = filter === 'all' || category.indexOf(filter) !== -1;
          item.style.display = show ? 'block' : 'none';
        });
      });
    });
  }

  var lightbox = document.querySelector('[data-lightbox]');
  var lightboxImage = document.querySelector('[data-lightbox-image]');
  var lightboxClose = document.querySelector('[data-lightbox-close]');
  var portfolioButtons = document.querySelectorAll('.portfolio-item');

  function closeLightbox() {
    if (lightbox) {
      lightbox.classList.remove('open');
      lightbox.setAttribute('aria-hidden', 'true');
    }
  }

  if (lightbox && lightboxImage && portfolioButtons.length) {
    portfolioButtons.forEach(function (button) {
      button.addEventListener('click', function () {
        var src = button.getAttribute('data-full');
        var alt = button.getAttribute('data-alt') || 'Portfolio image';

        lightboxImage.src = src;
        lightboxImage.alt = alt;
        lightbox.classList.add('open');
        lightbox.setAttribute('aria-hidden', 'false');
      });
    });

    if (lightboxClose) {
      lightboxClose.addEventListener('click', closeLightbox);
    }

    lightbox.addEventListener('click', function (event) {
      if (event.target === lightbox) {
        closeLightbox();
      }
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') {
        closeLightbox();
      }
    });
  }
})();
