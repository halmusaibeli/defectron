"""
creating texture out of image
"""
import argparse
import os
import cv2

def main(args):
    source_image = os.path.abspath(args.source)
    output_image = os.path.join(os.path.abspath(args.output), 'output.png')
    output_texture = os.path.relpath(os.path.join(os.path.abspath(args.output), 'output_texture.png'))
    
    # generate texture
    os.system('img2texture {source_image} {output_texture} --overlap 0.5'.format(
        source_image = os.path.relpath(source_image),
        output_texture = os.path.relpath(os.path.join(os.path.abspath(args.output), 'output_texture.png'))
    ))

    # repeat images
    img = cv2.imread(output_texture)
    vconcat_img = img
    for i in range(0,args.repeat-1):
        vconcat_img = cv2.vconcat([vconcat_img, img])
    hconcat_img = vconcat_img
    for i in range(0,args.repeat-1):
        hconcat_img = cv2.hconcat([hconcat_img, vconcat_img])
    
    # store image
    #cv2.imshow('concatinated images', hconcat_img)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    cv2.imwrite(output_image, hconcat_img)

    print('process completed.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/texture/wear_texture.png', help='image source for texture')
    parser.add_argument('--output', type=str, default='./data/texture', help = 'output textue')
    parser.add_argument('--repeat', type=int, default=2, help='repeatition of image')
    #--source data\texture\texture_source1.png --output data\texture\output.png --repeat 5
    args = parser.parse_args()
    main(args)