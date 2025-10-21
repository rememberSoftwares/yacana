// Menu initialization function
function initializeMainNavMenu(templatePath = 'templates/tuto_nav.html') {
    // Load the menu template
    fetch(templatePath)
        .then(response => response.text())
        .then(html => {
            document.getElementById('menu').innerHTML = html;

            // Initialize menu event handlers after template is loaded
            var $menu = $('.menu'),
                $menu_openers = $menu.children('ul').find('.opener');

            $menu_openers.each(function() {
                var $this = $(this);

                $this.on('click', function(event) {
                    // Prevent default.
                    event.preventDefault();

                    // Toggle.
                    $menu_openers.not($this).removeClass('active');
                    $this.toggleClass('active');

                    // Trigger resize (sidebar lock).
                    $(window).triggerHandler('resize.sidebar-lock');
                });
            });

            // Fix paths based on current page
            var currentPath = window.location.pathname;
            var $links = $('.menu a[href^="../pages/"]');
            
            if (currentPath.endsWith('index.html') || currentPath.endsWith('/')) {
                $links.each(function() {
                    var href = $(this).attr('href');
                    $(this).attr('href', '/yacana/' + href.replace('../pages/', 'pages/'));
                });
            }
        });
}

// Function to initialize the page navigation menu. It scans for H2 content on the page.
function initializePageNavMenu() {
    // Get all h2 elements in the main content
    const h2Elements = document.querySelectorAll('#main .inner h2');
    if (h2Elements.length === 0) return;

    // Create menu items from h2 elements
    const menuItems = Array.from(h2Elements).map(h2 => {
        // Create a slug from the text content
        const slug = h2.textContent.toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/(^-|-$)/g, '');
        
        // Add the id to the h2 element if it doesn't have one
        if (!h2.id) {
            h2.id = slug;
        }
        
        return `<li><a href="#${h2.id}">${h2.textContent}</a></li>`;
    }).join('');

    // Load the template and insert the menu items
    fetch('templates/page_nav.html')
        .then(response => response.text())
        .then(html => {
            // Replace the placeholder with actual menu items
            const menuHtml = html.replace('<!-- Dynamic menu items will be inserted here -->', menuItems);
            
            // Find the page navigation container and insert the menu
            const pageNavContainer = document.getElementById('page-nav-container');
            if (pageNavContainer) {
                pageNavContainer.innerHTML = menuHtml;
            }
        });
}
