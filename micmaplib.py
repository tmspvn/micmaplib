import os
import sys
import json
import time
import imageio
import subprocess
import numpy as np
import nibabel as nib
import itertools as itt
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# from prepare_inputs import Subject

"""
Microstructure Mapping Lab supporting functions
"""

from scipy.ndimage import zoom  # %% Nifti operations


def quicknii(inimg, func, outimg="/path/newimg.nii.gz", *args, **kwargs):
    """
    "That's pure magic!" - Everyone using this function. \n
    Usage: power 2 of img.nii and save with newname
        quicknii('/path/to/img.nii[gz]', np.power, outimg=path/to/newimg.nii[.gz], 2) \n
    inimg : <str>, path to input volume \n
    func : <func>, function to apply to the new image, image:np.array must be the first input of the func \n
    outimg : <str, False, None>, how to return the output image
            <str> -> 'full/path/where/to/save/newimg.nii' and return <outimg:str> \n
            <None> -> OVERWRITE computation to inimg \n
            <False> -> return np.array \n
    *args,**kwargs : additional arguments for the input function \n
    Version 1.0.2, 18/01/22
    """
    if isinstance(inimg, nib.nifti1.Nifti1Image):
        img = inimg
    else:
        img = nib.load(inimg)

    aff, hdr = img.affine, img.header
    newimg = nib.Nifti1Image(func(img.get_fdata(), *args, **kwargs),
                             affine=aff, header=hdr)
    if outimg is None:
        nib.save(newimg, inimg)  # overwrite input img
        return inimg
    elif outimg is False:
        return newimg  # return np.array()
    else:
        nib.save(newimg, outimg)
        return outimg  # save new image and return path


def bytes_check(i0, i1):
    '''Compare byte by byte two images and return the bytes where they differ'''
    start_time = time.time()
    with open(i0, "rb") as im0:
        with open(i1, "rb") as im1:
            img0, img1 = im0.read(), im1.read()
    if len(img0) == len(img1):
        imgbool = [False if b[0] == b[1] else True for b in zip(img0, img1)]
        s, t, idx = np.sum(imgbool), len(imgbool), np.where(imgbool)[0].tolist()
        print(str(t - s), '/', str(t), 'bytes matched. ', s, 'bytes differ')
        print('Byte index: ', str(idx))
        print('Pos0:', [img0[i] for i in idx])
        print('Pos1:', [img1[i] for i in idx])
        print("Done in %s secs" % (time.time() - start_time))
    else:
        print('Files length is different')
    return


def gzip(file, gz):
    if gz == False:
        newname_file = file.replace('.gz', '')
        os.system(f"gzip -f -c -d {file} > {newname_file}")
    elif gz == True:
        os.system(f"gzip {file} -f")
        newname_file = f'{file}.gz'
    else:
        newname_file = None
    return newname_file


# %% Header operations
def compare_headers(*args, comp_bytes=False):
    """
    Print sequentially compare all combinations of couples of headers, affine and header bytes
    INPUT: 2 or more images as str
    comp_bytes: Do header bytes comparison, default suppress
    """
    c = color
    for comb in itt.combinations(args, 2):
        hdr_0, aff_0, name_0 = dict(nib.load(comb[0]).header), nib.load(comb[0]).affine, os.path.basename(comb[0])
        hdr_1, aff_1, name_1 = dict(nib.load(comb[1]).header), nib.load(comb[1]).affine, os.path.basename(comb[1])
        hdr_ref = nifti_fields()
        printonce = True
        for k in hdr_0.keys():
            if np.all(str(hdr_0[k]) != str(hdr_1[k])):
                # Set spaces
                s = 20
                space0 = ((s - 5) - len(k)) * ' '
                space1 = (s - len(str(hdr_0[k]))) * ' '
                space = (s - len(str(hdr_1[k]))) * ' '
                # Print filenames once
                if printonce:
                    print('\n', (s - 5 + 2) * ' ', c.CBLUE, name_0, c.ENDC, sep='')  # s -5 + 2 since len(', ') = 2
                    print((s - 5 + 2) * ' ', '|', (s + 1) * ' ', c.CRED, name_1, c.ENDC,
                          sep='')  # 30 + 4 since len(', ') = 2 * 2 = 4
                    print((s - 5 + 2) * ' ', '\u2193', (s + 1) * ' ', '\u2193', sep='')  # 45 + 6
                    printonce = False
                # Print the comparison
                hdr_0[k] = hdr_0[k].tolist()
                hdr_1[k] = hdr_1[k].tolist()
                # Make list properly readable when printed
                if (isinstance(hdr_0[k], list) and isinstance(hdr_1[k], list)) and (
                        len(hdr_0[k]) > 2 or len(hdr_1[k]) > 2):
                    for i in range(len(hdr_0[k])):
                        space1 = (s - len(str(hdr_0[k][i]))) * ' '
                        space = (s - len(str(hdr_1[k][i]))) * ' '
                        if i == 0:  # First row
                            print(k, space0[0:-1] + '[' + str(hdr_0[k][i]),
                                  space1 + '[' + str(hdr_1[k][i]), space + hdr_ref[k], sep=', ')
                        elif i == len(hdr_0[k]) - 1:  # Last row
                            print(' ' * len(k), space0 + str(hdr_0[k][i]) + ']',
                                  space1[0:-1] + str(hdr_1[k][i]) + ']', sep=', ')
                        else:
                            print(' ' * len(k), space0 + str(hdr_0[k][i]),
                                  space1 + str(hdr_1[k][i]), sep=', ')
                else:
                    print(k, space0 + str(hdr_0[k]), space1 + str(hdr_1[k]), space + hdr_ref[k], sep=', ')

        # Affine comparison
        if np.any(aff_0 != aff_1):
            print('\nAffine:\n')
            print(c.CBLUE, name_0, c.ENDC, '\n', aff_0)
            print(c.CRED, name_1, c.ENDC, '\n', aff_1)
            print(f'{name_0} - {name_1}', '\n', aff_0-aff_1)
        # Bytes comparison
        if comp_bytes:
            with open(comb[0], "rb") as im0, open(comb[1], "rb") as im1:
                im_0, im_1 = im0.read()[0:353], im1.read()[0:353]
                different_bytes = [i for i in range(0, 353) if im_0[i] != im_1[i]]
            print(f'\nHeader bytes differ in {len(different_bytes)}/348, index:\n {different_bytes}')
    print('https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/')


def cpheader(from_im, to_im, newimg=False, cpbytes=False):
    """Copy the header by replacing the data of to_im to from_im.
        By default, tries overwrites to_im, newimg='path' to save a new img"""
    # ToDo: test copy bytes
    if not cpbytes:
        to_im_name = to_im
        from_im = nib.load(from_im)
        to_im = nib.load(to_im)
        new_img = to_im.__class__(to_im.dataobj[:], from_im.affine, from_im.header)
        if newimg:
            new_img.to_filename(newimg)
        else:
            out = to_im_name
            os.rename(out, out.replace('.nii', '__TMP4HDRCOPY__.nii'))
            new_img.to_filename(out)
            os.remove(out.replace('.nii', '__TMP4HDRCOPY__.nii'))
    else:
        with open(from_im, "rb") as donor:
            if newimg:
                with open(newimg, "wb") as receiver:
                    receiver.seek(0)  # set bytes offset at 0
                    receiver.write(donor.read()[0:349])  # First 348 bytes are the header
                    receiver.seek(349)  # set bytes offset at 349
                    with open(to_im, "rb") as datadonor:
                        receiver.write(datadonor.read()[349:])  # write data from the 349 byte
            else:
                with open(to_im, "wb") as receiver:
                    receiver.seek(0)  # set bytes offset at 0
                    receiver.write(donor.read()[0:349])  # First 348 bytes are the header


# %% Naming BIDS
def BIDS(SUB, imgtype=None, dot='.nii.gz', **kwargs):
    if isinstance(SUB, Subject):
        bids = f'/sub-CHUV{SUB.id}_ses-{SUB.ses}'
    else:
        bids = f'/sub-CHUV{SUB[0]}_ses-{SUB[1]}'
    # Keywords:
    if "fromto" in kwargs:
        xfmspec = kwargs["fromto"]
        if len(xfmspec) != 3:
            raise KeyError('fromto=["from","to","mode"] where mode is one of: points,surface,image,sphere')
        bids += f'_from-{xfmspec[0]}_to-{xfmspec[1]}_mode-{xfmspec[2]}'
        # mode must be one of: points,surface,image,sphere
    else:
        for spec, val in kwargs.items():
            bids += f'_{spec}-{val}'

    # Suffixes
    if dot == None or dot == False:
        dot = ''
    if imgtype != None:
        bids += f'_{imgtype}{dot}'
    return bids


# %% Registration and skullstripping functions

def synthstrip(img):
    o_img = img.replace(".nii.gz", "_brain.nii.gz")
    os.system(f'sudo docker run '
              f'-v {os.path.dirname(img)}/:/tmp '
              f'freesurfer/synthstrip '
              f'-i /tmp/{os.path.basename(img)} '
              f'-o /tmp/{os.path.basename(img).replace(".nii.gz", "_brain.nii.gz")}')
    if os.path.isfile(o_img):
        return o_img
    else:
        raise FileNotFoundError('SynthStrip failed!')

def skullstrip(inimg, refimg=None, outimg="path/to/prefix", inspace='T1w', refspace='MNI', template_res=1,
               ncpu=1, BiasFieldCorr=True, debug=False):
    """
    :param refspace:
    :param debug: save intermediate files
    :param BiasFieldCorr: perform (N4) bias field correction
    :param inimg: <str> input image
    :param refimg: <list>, [T1w_template, T1w_template_mask]
                   path to template with and its brain mask.
                   False or None: uses MNI152_T1_1mm and first brain mask from FSL
    :param outimg: <str> path to the file and prefix
    :param ncpu: <int>
    :return: Path to extracted brain file
    """

    os.system(f'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={ncpu - 1}')
    start_time = time.time()
    if refimg == False or refimg is None:
        if template_res != 0.5 and template_res != 1 and template_res != 2:
            raise ValueError('ERROR: template_res must be 0.5, 1 or 2 (higher number = faster)')

        fsldir = subprocess.getoutput('echo $FSLDIR') + '/data/standard'
        refimg = f'{fsldir}/MNI152_T1_{template_res}mm.nii.gz'
        templatemask = f'{fsldir}/MNI152_T1_1mm_brain_mask.nii.gz'
        # fill holes in the mask
        refmask = outimg + "_space-MNI_brainmask.nii.gz"
        os.system(f'fslmaths {templatemask} -fillh {refmask}')
    else:
        refimg, refmask = refimg[0], refimg[1]

    # Perform bias field correction
    itexist = False
    if BiasFieldCorr:
        print('N4BiasFieldCorrection')
        os.system(f'N4BiasFieldCorrection -d 3 -i {inimg} -s 2 -o {outimg + "_desc-N4BiasFieldCorr_T1w.nii.gz"}')
        inimg = outimg + f"_desc-N4BiasFieldCorr_T1w.nii.gz"
        itexist = True
    else:
        if os.path.isfile(outimg + f"_desc-N4BiasFieldCorr_T1w.nii.gz"):
            itexist = True
            inimg = outimg + f"_desc-N4BiasFieldCorr_T1w.nii.gz"

    # T1w -> T1template
    print('antsRegistration-based skullstrip')
    os.system(_ANTs_3stages_reg_skullstrip(outimg, refimg, inimg, verbose=debug))

    # Outputs of the previous step
    mat = outimg + '0GenericAffine.mat'
    inv_warp = outimg + '1InverseWarp.nii.gz'
    warp = outimg + '1Warp.nii.gz'
    in2outspace = outimg + 'Warped.nii.gz'
    inv_in2outspace = outimg + 'InverseWarped.nii.gz'

    # Bring back to original space the template brain mask
    # T1template_mask -> T1w
    outmask = outimg + f"_space-{inspace}_brainmask.nii.gz"
    flags = ['-d 3',  # dimensionality
             f'-e 3',  # input data type (3 = time-series)
             f'-i {refmask}',  # input filename
             f'-o {outmask}',  # output warped mask
             f'-r {inimg}',  # reference
             f'-t [{mat}, 1]',  # affine+rigid, inverse
             f'-t {inv_warp}']  # inverse warp
    if debug:
        print('antsApplyTransforms ' + ' '.join(flags))
    os.system('antsApplyTransforms ' + ' '.join(flags))

    # Concatenate trasform to save:
    T12MNI_xfm = f'{outimg}_from-{inspace}_to-{refspace}_mode-image_xfm.nii.gz'
    flags = f'-d 3 -e 3 -i {refmask} -o [{T12MNI_xfm}, 1] -r {inimg} -t [{mat}, 0] -t {warp}'
    os.system('antsApplyTransforms ' + flags)
    MNI2T1_xfm = f'{outimg}_from-{refspace}_to-{inspace}_mode-image_xfm.nii.gz'
    flags = f'-d 3 -e 3 -i {inimg} -o [{MNI2T1_xfm}, 1] -r {refmask} -t [{mat}, 1] -t {inv_warp}'
    os.system(f'antsApplyTransforms {flags}')

    # Apply the mask to original img
    def Fx(x):
        return x * (np.squeeze(nib.load(outmask).get_fdata()) > 0)

    outimg = quicknii(inimg, Fx, outimg=outimg + f"_space-{inspace}_brain.nii.gz")

    # Do not remove intermediate files
    if not debug:
        os.system(f'rm {mat} {inv_warp} {warp} {in2outspace} {inv_in2outspace} {refmask}')

    outputs = [outimg, outmask, T12MNI_xfm, MNI2T1_xfm]
    if BiasFieldCorr or itexist:
        outputs += [inimg]

    print("Brain extracted in: %s minutes" % ((time.time() - start_time) / 60))
    return outputs


def _ANTs_ResampleImge(inimg, outimg='path/name', shape=None, size_or_spacing=1, interp=4, dim=3):
    """Usage: ResampleImage imageDimension inputImage outputImage MxNxO [size=1,spacing=0] [interpolate type] [
    pixeltype]"""
    if shape is None:
        raise 'ERROR: shape must be a 3D list, e.g.: [200, 200, 200]'
    if size_or_spacing != 1 and size_or_spacing != 0:
        raise 'ERROR: size or spacing must be 1 for size, 0 for spacing'

    shape_ = 'x'.join([str(dim) for dim in shape])
    os.system(f'ResampleImage {dim} {inimg} {outimg} {shape_} {size_or_spacing} {interp} 6')
    return outimg


def _ANTs_3stages_reg_skullstrip(outputPrefix, fixedImage, movingImage, verbose=False):
    """
    antsRegistration call for hard-to-extract brains. Uses mask only for non-linear registration.
    :param outputPrefix: path/to/prefix
    :param fixedImage: path/to/nii
    :param movingImage: path/to/nii
    :return: The call as string (takes 10-15min to register)
    """
    call = [f"antsRegistration --verbose {int(verbose)} --random-seed 1 --dimensionality 3 --float 0 ",
            f"--collapse-output-transforms 1 ",
            f"--output [ {outputPrefix},{outputPrefix}Warped.nii.gz,{outputPrefix}InverseWarped.nii.gz ] ",
            f"--interpolation Linear ",
            f"--use-histogram-matching 0 ",
            f"--winsorize-image-intensities [ 0.005,0.995 ] ",
            f' --initial-moving-transform [ {fixedImage},{movingImage},1 ] ',
            f"",
            f"--transform Rigid[ 0.1 ] ",
            f"--metric MI[ {fixedImage},{movingImage},1,32,Regular,0.50 ] ",
            f"--convergence [ 1000x500x250x0,1e-7,50 ] ",
            f"--shrink-factors 8x4x2x1 ",
            f"--smoothing-sigmas 3x2x1x0vox ",
            f"",
            f"--transform Affine[ 0.1 ] ",
            f"--metric MI[ {fixedImage},{movingImage},1,32,Regular,0.50 ] ",
            f"--convergence [ 1000x500x250x100,1e-7,50 ] ",
            f"--shrink-factors 8x4x2x1 ",
            f"--smoothing-sigmas 3x2x1x0vox ",
            f"",
            f"--transform SyN[ 0.2,3,0 ] ",
            f"--metric CC[ {fixedImage},{movingImage},1, 1, Regular,0.25 ] ",
            f"--convergence [ 2000x1000x1000x300x100,1e-10,100 ] ",
            f"--shrink-factors 16x8x4x2x1 ",
            f"--smoothing-sigmas 3x3x2x1x0vox "]
    return ' '.join(call)


def _antsApplyTransformToPoints(surface, trasforms=[], space='B0'):
    # Convert surface to csv
    surf = nib.load(surface)
    coords, triangles = surf.agg_data()
    surface_csv = 'original_surface.csv'
    np.savetxt(surface_csv, np.concatenate([coords, np.zeros([coords.shape[0], 1])], axis=1),
               header='x,y,z,t', comments='', delimiter=',')
    # Set up apply
    warped_csv = 'warped_surface.csv'
    ts = [f'-t {t} ' for t in trasforms]
    cmd = f'antsApplyTransformsToPoints -d 3 -p 1 -i {surface_csv} -o {warped_csv} {" ".join(ts)}'
    os.system(cmd)
    # Remove surface csv
    os.remove(surface_csv)
    warped_coords = np.loadtxt(warped_csv, skiprows=1, delimiter=',')
    # assign new coords (coord array idx = 0, triangle = 1) to surface
    surf.darrays[0].data = warped_coords[:, 0:3]
    warped_surface = surface.replace("hemi", f"space-{space}_hemi")
    nib.save(surf, warped_surface)
    return warped_surface


def add_FScras_to_surfaces(surface, talairachlta='/mri/transforms/talairach.lta'):
    # Add offset to FS surfaces for the mm space
    # https://neurostars.org/t/freesurfer-cras-offset/5587
    # CRAS = Central voxel to the RAS center of the image

    surf = nib.load(surface)
    with open(talairachlta, 'r') as fp:
        cras = [np.fromstring(l[9:-1], sep=' ') for l in fp if 'cras' in l][0]
    # offset by cras the surfaces coordinates
    print('Offsetting:', surface)
    print('CRAS: ', cras)
    surf.darrays[0].data += cras
    nib.save(surf, surface)

    return surface


def quickAffine(f, m, o, debug=False, metric='MI', conv='6', itsc=1, sz=None):
    """antsQuickReg: f=fixed, m=moving, o=outPrefix, conv=convergence thr, itsc=base number itererations * scale"""
    if metric == 'CC':
        rb = 1  # radius 1
    else:
        rb = 32  # bins
    if sz is None:
        sz = [0.1, 0.1]
    l0, l1, l2, l3, rep = 1000 * itsc, 500 * itsc, 250 * itsc, 100 * itsc, 10 * int(1 + itsc / 2)
    call = [f'antsRegistration ',
            f' --verbose {int(debug)} ',
            f' --dimensionality 3 ',
            f' --float 0 ',
            f' --collapse-output-transforms 1',
            f' --output [ {o},{o}Warped.nii.gz,{o}InverseWarped.nii.gz ] ',
            f' --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ] ',
            f' --initial-moving-transform [ {f},{m},1 ] ',
            f' --transform Rigid[ {sz[0]} ] ',
            f' --metric {metric}[ {f},{m},1,{rb},Regular,0.5 ] ',
            f' --convergence [ {l0}x{l1}x{l2}x{l3},1e-{conv},{rep} ] ',
            f' --shrink-factors 8x4x2x1 ',
            f' --smoothing-sigmas 3x2x1x0vox ',
            f' --transform Affine[ {sz[1]} ] ',
            f' --metric {metric}[ {f},{m},1,{rb},Regular,0.5 ] ',
            f' --convergence [ {l0}x{l1}x{l2}x{l3},1e-{conv},{rep} ] ',
            f' --shrink-factors 8x4x2x1 ',
            f' --smoothing-sigmas 3x2x1x0vox']
    if debug:
        print(call)
    return os.system(' '.join(call))


def quickWarp(f, m, o, debug=False, metric='MI', conv=6, itsc=1, rdsc=1, smosc=1, sz=[0.1, 0.1, 0.2]):
    """antsQuickReg: f=fixed, m=moving, o=outPrefix, conv=convergence thr, itsc=base number itererations * scale"""
    if metric == 'CC':
        rb = 1 * int(rdsc)  # radius
    else:
        rb = 32  # bins
    itsc = int(itsc)
    l0, l1, l2, l3, rep = 1000 * itsc, 500 * itsc, 250 * itsc, 100 * itsc, 10 * int(1 + itsc / 2)
    s0, s1, s2, s3, = int(3 * smosc), int(2 * smosc), int(1 * smosc), int(0 * smosc)
    call = [f'antsRegistration ',
            f' --verbose {int(debug)} ',
            f' --dimensionality 3 ',
            f'--collapse-output-transforms 1',
            f' --float 0 ',
            f' --collapse-output-transforms 1',
            f' --output [ {o}, {o}Warped.nii.gz,{o}InverseWarped.nii.gz ] ',
            f' --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ] ',
            f' --initial-moving-transform [ {f},{m},1 ] ',
            f' --transform Rigid[ {sz[0]} ] ',
            f' --metric {metric}[ {f},{m},1,{rb},Regular,0.25 ] ',
            f' --convergence [ {l0}x{l1}x{l2}x{l3},1e-{conv},{rep} ] ',
            f' --shrink-factors 8x4x2x1 ',
            f' --smoothing-sigmas 3x2x1x0vox ',
            f' --transform Affine[ {sz[1]} ] ',
            f' --metric {metric}[ {f},{m},1,{rb},Regular,0.25 ] ',
            f' --convergence [ {l0}x{l1}x{l2}x{l3},1e-{conv},{rep} ] ',
            f' --shrink-factors 8x4x2x1 ',
            f' --smoothing-sigmas 3x2x1x0vox',
            f" --transform SyN[ {sz[2]},3,0 ] ",
            f" --metric {metric}[ {f},{m},1,{rb}, Regular,0.25 ] ",
            f" --convergence [{l0}x{l1}x{l2}x{l3},1e-{conv + 1},{rep}]",
            f" --shrink-factors 8x4x2x1 ",
            f" --smoothing-sigmas {s0}x{s1}x{s2}x{s3}vox "]
    if debug:
        print(call)
    return os.system(' '.join(call))


def _ANTs_ApplyTransform(indwi, warpeddwi, reference=None, *args, interp=3, debug=False, antsbin=''):
    e = len(nib.load(indwi).shape) - 1
    if e == 3:
        e = '-e 3'  # 3D images are scalars for ANTS
    else:
        e = ''
    interpolation = {0: 'Linear', 1: 'NearestNeighbor', 2: 'GenericLabel', 3: 'BSpline[3]'}
    spec = f' --default-value 0 --interpolation {interpolation[interp]} '
    call = f'{antsbin}antsApplyTransforms -d 3 {e} -i {indwi} -o {warpeddwi} -r {reference}'
    for trasform in args:
        call += f' -t {trasform}'
    call = call + f'{spec}'
    if debug:
        print(f'\n{call}\n')
    return os.system(call)


def _invertAffine(affine):
    return f'[{affine}, 1]'

# %% topup & EDDY stuff

def eddy(dwi, bval, bvec, mask, outf_ses, bids_prefix, acqp_path, indextxt_path, topup,
         apply_eddy_params=None, save_displacement_fields=True, openmp=True):
    start = time.time()

    if not os.path.exists(f'{outf_ses}/eddy'):
        os.makedirs(f'{outf_ses}/eddy')

    if apply_eddy_params:
        if not os.path.exists(apply_eddy_params):
            if not os.path.exists(f'{outf_ses}/eddy/eddy.eddy_parameters'):
                raise FileNotFoundError('pre-computed correction not found.')
            else:
                apply_eddy_params = f'{outf_ses}/eddy/eddy.eddy_parameters'

    eddyout = f'{outf_ses}/eddy'

    if openmp:
        eddytype = 'eddy_openmp'
    else:
        eddytype = 'eddy'

    dfields = ''
    if save_displacement_fields:
        dfields = f"--dfields"

    if apply_eddy_params:
        print(f"Apply pre-computed Eddy parameters from: {apply_eddy_params}")
        opt = f"--niter=0 --init={apply_eddy_params}"
    else:
        print(f"Eddy Correction")
        opt = f"--niter=8 --fwhm=10,6,4,2,0,0,0,0"

    call = [f"{eddytype}",
            f"--imain={dwi}",
            f"--mask={mask}",
            f"--index={indextxt_path}",
            f"--acqp={acqp_path}",
            f"--bvecs={bvec}",
            f"--bvals={bval}",
            f"--topup={topup}",
            f"--out={eddyout}/eddy",
            opt,
            f"--fep",
            dfields,
            f"--data_is_shelled"]
    os.system(' '.join(call))
    print("Done")
    end = time.time()
    print(f"In :{int((end - start) // 3600)}h, {int((end - start) % 3600 // 60)}min and {int((end - start) % 60)}s.")

    # Rename in BIDS standards
    os.rename(f"{outf_ses}/eddy/eddy.nii.gz", f"{outf_ses}/{bids_prefix}_desc-eddy_dwi.nii.gz")

    if not os.path.isfile(f'{outf_ses}/{bids_prefix}_desc-eddy_dwi.nii.gz'):
        raise FileNotFoundError('Eddy Failed!')

    return f"{outf_ses}/{bids_prefix}_desc-eddy_dwi.nii.gz"


# %% DKI FIT


def dkifit(dwi, bval, bvec, mask, outdir, MATLABEXEC, constraints=[0, 60, 0], detect_outlier=False, sigmamap=False):
    mmlpath = get_mml_cwd()
    if not os.path.isfile(MATLABEXEC):
        raise FileNotFoundError('ERROR: cannot find matlab executable')
    elif not os.path.isfile(dwi):
        raise FileNotFoundError(f'ERROR: cannot find dwi: {dwi}')
    elif not os.path.isfile(bval):
        raise FileNotFoundError(f'ERROR: cannot find bval: {bval}')
    elif not os.path.isfile(bvec):
        raise FileNotFoundError(f'ERROR: cannot find bvec: {bvec}')
    elif not os.path.isfile(mask):
        raise FileNotFoundError(f'ERROR: cannot find mask: {mask}')
    elif not os.path.isdir(f'{mmlpath}/dkiprocessor'):
        raise NotADirectoryError(f'ERROR: cannot find dkiprocessor master directory in: {mmlpath}')

    # extract from.gz the images
    # os.system(f"gzip -f -d {dwi}")
    # os.system(f"gzip -f -d {mask}")
    ungzip(dwi)
    ungzip(mask)
    masknii = mask.replace('.gz', '')
    newname_dwi = dwi.replace('.gz', '')

    if detect_outlier:
        os.system(f"gzip -f -d {sigmamap}")
        newname_sigmamap = sigmamap.replace('.gz', '')
        sigmamap = f", '{newname_sigmamap}'"
        print('DKI fit + outliers detection')
    else:
        print('DKI fit')
        sigmamap = ''

    a = '"'  # apostrophe
    os.system(f"{MATLABEXEC}"
              f" -sd {os.getcwd()} -batch {a}"
              f"addpath(genpath('{mmlpath}/rpgdegibbs/lib'));"
              f"addpath(genpath('{mmlpath}/NIFTI'));"
              f"addpath(genpath('{mmlpath}/dkiprocessor'));"
              f"addpath(genpath('{mmlpath}/dkiprocessor/supporting_functions'));"
              f"addpath(genpath('{mmlpath}'));"
              f"disp('...'); "
              f"run_dkiprocessor('{newname_dwi}',"
              f"'{bval}', '{bvec}',"
              f"'{masknii}',"
              f"'{outdir + '/'}',"
              f"{str(constraints)}, {int(detect_outlier)} {sigmamap});"
              f"quit;{a}")
    # compress back
    os.system(f"gzip {masknii} -f")
    os.system(f"gzip {newname_dwi} -f")
    if detect_outlier:
        os.system(f"gzip {newname_sigmamap} -f")
    # Check outputs
    if not os.path.isfile(outdir + '/dt_wlls.mat'):
        raise FileNotFoundError('ERROR: KDI fit was not computed because tensor mat cannot be found')

    # Check if all file exist
    # DTI
    out_list = ['md', 'ad', 'rd', 'fa', 'fefa', 'mk', 'rk', 'ak']
    list_nii = [os.path.isfile(outdir + f'/{img}.nii') for img in out_list]
    if np.array(list_nii).all():
        # Compress them
        [os.system(f"gzip {outdir}/{img}.nii -f") for img in out_list]
        list_gz = [outdir + f'/{img}.nii.gz' for img in out_list]
    else:
        raise FileNotFoundError('ERROR: Fit incomplete, maps are missing')

    # make sidecar json
    sidecar = {"DTI": {'fitMethod': 'WLS'}, 'DKI': {'fitMethod': 'WLS'}}
    file = open(outdir + '/sidecar.json', "w")
    json.dump(sidecar, file, indent=4, separators=(",", ": "))
    file.close()

    return list_gz


# %% Bvals operations


def id_shell(bvals, thr=250):
    # bvals = np.loadtxt(bval, dtype=int)
    unique_shells = []
    bvalue = [np.unique(bvals)[0]]
    shells_idx = np.zeros_like(bvals)
    shells_idx_onsets = np.zeros_like(bvals)
    c = 0
    for i, v in enumerate(bvals[1::]):
        if v - np.mean(bvalue) < thr:
            bvalue += [int(v)]
        else:
            unique_shells += [bvalue]
            bvalue = [int(v)]
            c += 1
            shells_idx_onsets[i] = True
        shells_idx[i + 1] = c

    rounded_shells = [int(np.rint(np.mean(sh) / 100) * 100) for sh in unique_shells]
    rounded_shells = [np.unique(sh)[0] for sh in rounded_shells]
    unique_shells = [np.unique(sh) for sh in unique_shells]

    return unique_shells, rounded_shells, shells_idx, shells_idx_onsets


# %% PLOTTING FUNCTIONS

def mkgif(img, path=False, view=0, slice4d=False, rotate=False, rotaxes=(1, 2), flip=False, rewind=True, winsorize=[1, 98],
          timebar=True, crosshair=False, scale=2, cmap=False, crop=True, vol_wise_norm=False, fps=60, concat_along=1):
    """
    :param img: image filepath, nibNifti1Image or ndarray
    :param path: string, filename to save the .gif. if False use input filename. [default=False]
    :param view: string or bool, specify type of view to plot: 'sagittal' or 0 [Default], 'coronal' or 1, 'axial'or 2
    :param rewind: bool, repeat the animation backwards. [default:=True]
    :param winsorize: list, winsorize image intensities to remove extreame values. [default:=[1,98]]
    :param rotaxes: tuple, axis along which rotate the image. [default=(1, 2)]
    :param rotate: int [1:3], int*90-degree rotation counter-clockwise. [default=False]
    :param slice4d: int: index where to slice chosen view in 4D image. [default=False -> img.shape[1]//2]
    :param timebar: bool, print timebar at the bottom [default=True]
    :param flip: int, axis to flip [default=False]
    :param crosshair: list [x, y], print crosshair at coordinates. [default=False]
    :param scale: int, factor to linear interpolate the input, time dimension is not interpolated. [default=2]
    :param cmap: str, add matplotlib cmap to image, eg: 'jet'. [default=2]
    :param crop: bool, crop air from image. [default=True]
    :param vol_wise_norm: normalize image volume-wise, only if timeseries. [default=False]
    :param fps: int, frame per second. Max 60. [default=60]
    :param concat_along: concatenate multiple images along a specific axis. [default=1]
    return file path
    todo: add plot all 3 views in one command
    """
    # Iterates through the first axis, collapses the last if ndim ==4
    # [ax0, ax1, ax2, ax3] == [Sagittal, Coronal, Axial, time] == [X, Y, Z, T]
    if not isinstance(img, list):
        imgsl = [img]
    else:
        imgsl = img

    toconcat = []
    for img in imgsl:
        if isinstance(img, str):
            inputimg = img
            img = nib.load(img).get_fdata()
        elif isinstance(img, nib.nifti1.Nifti1Image):
            inputimg = img.get_filename()
            img = img.get_fdata()
        elif isinstance(img, np.ndarray):
            if not path:
                raise IsADirectoryError("ERROR: when using a ndarray you must specify an output filename")
            img = img

        # Crop air areas
        if crop:
            if img.ndim == 4:
                xv, yv, zv = np.nansum(img, axis=(1, 2, 3)) > 0, \
                             np.nansum(img, axis=(0, 2, 3)) > 0, \
                             np.nansum(img, axis=(0, 1, 3)) > 0
                img, img, img = img[xv, :, :, :], \
                                img[:, yv, :, :], \
                                img[:, :, zv, :]
            else:
                xv, yv, zv = np.nansum(img, axis=(1, 2)) > 0, \
                             np.nansum(img, axis=(0, 2)) > 0, \
                             np.nansum(img, axis=(0, 1)) > 0
                img, img, img = img[xv, :, :], \
                                img[:, yv, :], \
                                img[:, :, zv]

        viewsstr = {'sagittal': 0, 'coronal': 1, 'axial': 2}
        views = {0: [0, 1, 2], 1: [2, 0, 1], 2: [1, 2, 0]}  # move first the dimension to slice for chosen view
        if isinstance(view, str):
            view = viewsstr[view]
        if img.ndim == 4:
            img = np.moveaxis(img, [0, 1, 2, 3], views[view] + [3])
        else:
            img = np.moveaxis(img, [0, 1, 2], views[view])

        if rotate:
            img = np.rot90(img, k=rotate, axes=rotaxes)  # Rotate along 2nd and 3rd axis by default

        if img.ndim == 4:
            img = np.moveaxis(img, 3, 0)  # push 4th dim (time) first since imageio iterates the first
            if isinstance(slice4d, bool):  # Then slice 2nd dimension to allow 3D animation
                img = img[:, img.shape[1] // 2, :, :]
            else:
                img = img[:, slice4d, :, :]

        # Winsorize and normalize intensities for plot
        Lpcl, Hpcl = np.nanpercentile(img, winsorize[0]), np.nanpercentile(img, winsorize[1])
        img[img < Lpcl], img[img > Hpcl] = Lpcl, Hpcl
        img = img * 255.0 / np.nanmax(img)
        if vol_wise_norm:  # normalize volume-wise
            img = np.array([img[idx, ...] * 255.0 / np.nanmax(img[idx, ...]) for idx in range(img.shape[0])])

        if not isinstance(flip, bool):
            img = np.flip(img, axis=flip)  # flip a dim if needed

        if not isinstance(scale, bool):  # interpol view but no time
            img = np.array([zoom(img[idx, ...], scale, order=1) for idx in range(img.shape[0])])
        else:
            scale = 1  # no interpol

        # timebar
        if timebar:
            tres = img.shape[2] / img.shape[0]
            it = 0
            for i in range(img.shape[0]):
                it += tres
                img[i, img.shape[1] - 1, 0:int(it)] = 255  # [i,0,0] is upper left corner

        # crosshair
        if isinstance(crosshair, list):
            xmask, ymask = np.zeros_like(img, dtype=bool), np.zeros_like(img, dtype=bool)
            xmask[:, crosshair[0] * scale, :], ymask[:, :, crosshair[1] * scale] = True, True
            mask = np.logical_xor(xmask, ymask)
            img[mask] = 255

        # repeat the animation backwards
        if rewind:
            img = np.concatenate([img, np.flip(img, axis=0)], axis=0)

        # set cmap
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
            # return warning 'Convert image to uint8 prior to saving' but if cast it breaks
            img = cmap(img.astype(np.uint8))
        else:
            img = img.astype(np.uint8)

        # store
        toconcat += [img]

    # set outputpath if not specified
    if not path:
        path = inputimg.replace('.nii.gz', '.gif')

    # concatenate images to plot
    img = np.concatenate(toconcat, axis=concat_along)
    # write gif
    imageio.mimwrite(path, img, fps=fps)
    return path


# %% ADDITIONAL FUNCTIONS

def getlistimg(dir=os.getcwd()):
    """Get a convenient list of ONLY images (.nii, .nii.gz) in a folder ToDo: refractor to getlistnii()"""
    if not os.path.isdir(dir):
        raise FileNotFoundError('Input is not a directory')
    o = [f'{dir}/{k}' for k in os.listdir(dir) if '.nii' in k]
    o.sort()
    return o


def get_mml_cwd():
    """Get current working directory of micmaplib.py"""
    return os.path.dirname(__file__)


def ungzip(file):
    newname_dwi = file.replace('.gz', '')
    os.system(f"gzip -f -c -d {file} > {newname_dwi}")


def nifti_fields():
    """return dict with all field description from: https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h """
    fields = {'sizeof_hdr': 'int, MUST be 348',
              'data_type': 'char, ++UNUSED++',
              'db_name': 'char, ++UNUSED++',
              'extents': 'int, ++UNUSED++',
              'session_error': 'char, ++UNUSED++',
              'regular': 'char, ++UNUSED++',
              'dim_info': 'char, MRI slice ordering',
              'dim': 'short, Data array dimensions',
              'intent_p1': 'float, 1st intent parameter',
              'intent_p2': 'float, 2nd intent parameter',
              'intent_p3': 'float, 3rd intent parameter',
              'intent_code': 'short, NIFTI_INTENT_* code',
              'datatype': 'short, Defines data type!',
              'bitpix': 'short, Number bits/voxel',
              'slice_start': 'short, First slice index',
              'pixdim': 'float, Grid spacings',
              'vox_offset': 'float, Offset into .nii file',
              'scl_slope': 'float, Data scaling: slope',
              'scl_inter': 'float, Data scaling: offset',
              'slice_end': 'short, Last slice index',
              'slice_code': 'char, Slice timing order',
              'xyzt_units': 'char, Units of pixdim[1..4]',
              'cal_max': 'float, Max display intensity',
              'cal_min': 'float, Min display intensity',
              'slice_duration': 'float, Time for 1 slice',
              'toffset': 'float, Time axis shift',
              'glmax': 'int, ++UNUSED++',
              'glmin': 'int, ++UNUSED++',
              'descrip': 'char, any text you like',
              'aux_file': 'char, auxiliary filename',
              'qform_code': 'short, code{0:arbitrary,1:scanner,2:anatomical,3:talairach,4:MNI}',
              'sform_code': 'short, code{0:arbitrary,1:scanner,2:anatomical,3:talairach,4:MNI}',
              'quatern_b': 'float, Quaternion b param',
              'quatern_c': 'float, Quaternion c param',
              'quatern_d': 'float, Quaternion d param',
              'qoffset_x': 'float, Quaternion x shift',
              'qoffset_y': 'float, Quaternion y shift',
              'qoffset_z': 'float, Quaternion z shift',
              'srow_x': 'float, 1st row affine transform',
              'srow_y': 'float, 2st row affine transform',
              'srow_z': 'float, 3st row affine transform',
              'intent_name': "char, 'name' or meaning of data",
              'magic': 'char, MUST be "ni1\\0" or "n+1\\0"'}
    return fields


class color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
