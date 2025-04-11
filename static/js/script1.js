document.addEventListener("DOMContentLoaded", function() {
    const flipbook = document.getElementById('flipbook');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const backgroundMusic = document.getElementById('backgroundMusic');
    const volumeBtn = document.getElementById('volumeBtn');
    
    // Set total images (you can modify this based on your needs)
    const totalImages = 5; // or parseInt(value) if value is properly set
    let currentImageIndex = 0;
    
    // Function to update the current image
    function updateImage(index) {
        if (index < 0 || index >= totalImages) return;
        
        const imageContainer = flipbook.querySelector('.page');
        if (!imageContainer) return;
        
        const img = imageContainer.querySelector('img') || new Image();
        img.src = `/static/image/${index + 1}.png`;
        img.alt = `Page ${index + 1}`;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        
        // Add error handling for image loading
        img.onerror = function() {
            console.error(`Failed to load image: ${img.src}`);
            // Show error message in the container
            imageContainer.innerHTML = `
                <div style="color: red; padding: 20px;">
                    Failed to load image ${index + 1}. 
                    Tried path: ${img.src}
                </div>
            `;
        };
        
        img.onload = function() {
            console.log(`Successfully loaded image: ${img.src}`);
        };
        
        if (!imageContainer.contains(img)) {
            imageContainer.innerHTML = '';
            imageContainer.appendChild(img);
        }
    }
    
    // Navigation event listeners
    prevBtn.addEventListener('click', function() {
        if (currentImageIndex > 0) {
            currentImageIndex--;
            updateImage(currentImageIndex);
        }
    });
    
    nextBtn.addEventListener('click', function() {
        if (currentImageIndex < totalImages - 1) {
            currentImageIndex++;
            updateImage(currentImageIndex);
        }
    });
    
    // Volume control
    volumeBtn.addEventListener('click', function() {
        if (backgroundMusic.paused) {
            backgroundMusic.play().catch(e => console.warn('Error playing background music:', e));
            volumeBtn.classList.add('volume-on');
        } else {
            backgroundMusic.pause();
            volumeBtn.classList.remove('volume-on');
        }
    });
    
    // Debug information
    console.log('Current directory:', window.location.href);
    console.log('Attempting to load first image at:', '/static/image/1.png');
    
    // Initialize with first image
    updateImage(0);
});