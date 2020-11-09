n_images = 1000;
magnification = 60;
output_dir = '/home/koosk/data/images/adipocyte/generated_masks/';

copyfile(['simcep_options', num2str(magnification), '.m'], 'simcep_options.m')

parfor i = 1:n_images
    [~, binary] = simcep;
    fname = ['generated_nuclei_mask_', num2str(magnification), 'x-', datestr(now, 'YYYY-mm-DD_HHMMssFFF'), '.png'];
    fpath = fullfile(output_dir, fname);
    imwrite(binary, fpath);
end

delete('simcep_options.m')