# 1. split tiff
tiffsplit  *.tif

# 2. Use shotwell to enhance and gthumb to see where to crop

# 3. crop imgs
mogrify -crop 120x120+161+86  x*.tif

# 4. imgs2vid
ffmpeg -r 1 -pattern_type glob -i 'x*.tif' -c:v libx264 out.mp4
