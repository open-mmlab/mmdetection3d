[33mcommit 962f093736ffe55c089bc618842a8b8567318c8c[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mdev-1.x[m[33m, [m[1;31morigin/dev-1.x[m[33m)[m
Author: chbehrens <chbehrens@users.noreply.github.com>
Date:   Wed Jan 17 09:23:50 2024 +0100

    [Fix] Replace np.long with np.int64 (#2872)
    
    Co-authored-by: Christian Behrens <9531979+chbehrens@users.noreply.github.com>

[33mcommit 0ef13b83fac1eaad626c9989feec6b1a0b2588e7[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Mon Jan 8 15:22:00 2024 +0800

    Bump to v1.4.0 (#2866)

[33mcommit 88b86943b550957ffbc838dc63d455c5ead2a1d3[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Thu Jan 4 21:19:29 2024 +0800

    [Feature] Support PGD and multi-view FCOS3D++ on Waymo  (#2835)
    
    Co-authored-by: JingweiZhang12 <zjw18@mails.tsinghua.edu.cn>
    Co-authored-by: sjh <sunjiahao1999>

[33mcommit 2dad86c23fed102e7bf107b41b13e9363d4704e4[m
Author: YirongYan <yanyirong@sjtu.edu.cn>
Date:   Thu Jan 4 18:54:39 2024 +0800

    [Feature]Support NeRF-Det (#2732)

[33mcommit 8deeb6e203f80329f33759c9e5a2ef6cb07902d3[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Thu Jan 4 12:08:14 2024 +0800

    [Fix] fix MinkUNet & SPVCNN configs (#2854)

[33mcommit 762e3b53ca30e4cb0302a40330a609216236e541[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Thu Dec 28 21:34:59 2023 +0800

    [Feature] Support DSVT training (#2738)
    
    Co-authored-by: JingweiZhang12 <zjw18@mails.tsinghua.edu.cn>
    Co-authored-by: sjh <sunjiahao1999>

[33mcommit 5b88c7b8352f6cddf3cfdc47fa37e0fa0812bacc[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Wed Dec 27 17:15:55 2023 +0800

    [Refactor] Refactor Waymo dataset_converter/dataset/evaluator (#2836)
    
    Co-authored-by: sjh <sunjiahao1999>

[33mcommit 395b86d7d6f619fb8dd176a4d63f9c1755e189df[m
Author: Zephyr <zephyr_xf@qq.com>
Date:   Wed Dec 13 16:08:02 2023 +0800

    Dev 1.x Adapt PETR to mmcv v2 version dataset  (#2800)

[33mcommit 5c0613be29bd2e51771ec5e046d89ba3089887c7[m[33m ([m[1;33mtag: [m[1;33mv1.3.0[m[33m)[m
Merge: ac289b35 42e6dc5b
Author: Tai-Wang <tab_wang@outlook.com>
Date:   Thu Oct 19 15:29:23 2023 +0800

    Update version file and circleci for v1.3.0
    
    Update version file and circleci for v1.3.0

[33mcommit 42e6dc5b7b4d997b7a3d4019b4575d0399807694[m
Author: sunjiahao1999 <578431509@qq.com>
Date:   Thu Oct 19 15:15:26 2023 +0800

    fix resource_name

[33mcommit be3a6c7ccab46a2c2144417977993461a5d11d4b[m
Author: sunjiahao1999 <578431509@qq.com>
Date:   Thu Oct 19 15:13:09 2023 +0800

    fix version and resource_class & image name

[33mcommit ac289b35d23a0e921fa2782bb8d29a513f2b91c0[m
Merge: b0e8ece9 12b595ca
Author: Tai-Wang <tab_wang@outlook.com>
Date:   Thu Oct 19 14:34:40 2023 +0800

    Bump version to v1.3.0
    
    Bump version to v1.3.0

[33mcommit 12b595ca67a36469bd7df774d6d191fbde283791[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Thu Oct 19 14:02:55 2023 +0800

    [Fix] CI min gpu version and GitHub merge test (#2782)

[33mcommit 5a533e79b4635c2b62176d415a8ae2a38dab46f1[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Thu Oct 19 10:58:47 2023 +0800

    Bump to v1.3.0 (#2778)

[33mcommit db39fd4a046b9aee0551f594859eda919a68c430[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Wed Oct 18 20:49:22 2023 +0800

    [Enhance] Use Inferencer to implement Demo (#2763)

[33mcommit b0e8ece9ba0d76713b86ca4da4060d41bbc86451[m
Author: LRJKD <101466907+LRJKD@users.noreply.github.com>
Date:   Wed Oct 18 16:12:41 2023 +0800

    [Fix] Fix support for devices other than Cuda in FreeAnchor3DHead (#2769)

[33mcommit f4c032e41af079e8305b5ad183e6680505ba13df[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Wed Oct 18 16:10:40 2023 +0800

    [Fix] Delete unused __init__.py in mmdet3d/configs (#2773)

[33mcommit 9b77c74929c7c7983de9dcb491a4a26ea01cda66[m
Author: Xiang Xu <xuxiang0103@gmail.com>
Date:   Wed Sep 20 16:16:27 2023 +0800

    [Fix] Fix docs build error (#2739)

[33mcommit 6e6edfab1c45ed17a9a6a4dac3d5a02bc29b8f21[m
Author: Zhao Cake <zy9787101052763@126.com>
Date:   Wed Sep 13 16:44:57 2023 +0800

    [CodeCamp2023-516]Add new configuration files for Cylinder3D (#2681)

[33mcommit 48fd72ff04135fed379ba282e8265a6bbe9b912d[m
Author: Xiang Xu <xuxiang0103@gmail.com>
Date:   Wed Sep 13 16:39:24 2023 +0800

    [Fix] Fix potential bug in lasermix (#2710)

[33mcommit c642055f0b77edf925d63aaf606078c970ba1217[m
Author: zhengjie.xu <jerryxuzhengjie@gmail.com>
Date:   Wed Sep 13 03:08:36 2023 -0500

    [Doc]: Update QRcode (#2703)

[33mcommit 74878d1ea83fe0b4eab728e5eee0cd3340e1aacd[m
Author: 1uciusy <37854746+1uciusy@users.noreply.github.com>
Date:   Wed Sep 13 16:07:55 2023 +0800

    [Docs] Add BEV-based detection pipeline in NuScenes Dataset tutorial (#2672)
    
    * update the part of  in doc of nuScenes dataset
    
    * update nuScenes tutorial
    
    * add alternative bev sample code and necessary description for the nuscenes dataset
    
    * update nuscenes tutorial
    
    * update nuscenes tutorial
    
    * update nuscenes tutorial
    
    * use two subsections to introduce monocular and BEV
    
    * use two subsections to introduce monocular and BEV
    
    * use two subsections to introduce monocular and BEV
    
    * update NuScenes dataset BEV based tutorial
    
    * update NuScenes dataset BEV based tutorial

[33mcommit c04831c5d2c360dff8732d606204c94fc1eb8e7a[m
Author: Xiang Xu <xuxiang0103@gmail.com>
Date:   Tue Aug 1 19:40:58 2023 +0800

    [Fix] Update new config type (#2655)
    
    * update new config
    
    * fix bug

[33mcommit 5638af7d1d54fb6e4faaca9903e7afda3efbca57[m
Author: Xiang Xu <xuxiang0103@gmail.com>
Date:   Tue Aug 1 19:07:50 2023 +0800

    [CI] update torch version in ci (#2652)

[33mcommit 6e1d6a47b5edac9032577f8c9840b1371d5d0449[m
Author: Xiang Xu <xuxiang0103@gmail.com>
Date:   Tue Aug 1 19:07:02 2023 +0800

    [Feature] Support `CENet` in `Projects` (#2619)
    
    * add cenet in projects
    
    * add aux head for cenet
    
    * fix some potential bugs
    
    * update param_scheduler
    
    * finish the readme
    
    * remove redundant config
    
    * add fps

[33mcommit 5cd59f780bfc77fbf422001c3f5a42ed4582abb5[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Tue Aug 1 19:05:36 2023 +0800

    [Fix]: fix open3d version bug (#2633)

[33mcommit 5d822daa327461f6b1debdab0822fb74e4bda6a7[m
Author: Zhiqiang Wang <zhiqwang@foxmail.com>
Date:   Tue Aug 1 17:44:24 2023 +0800

    [Doc]fix broken links (#2650)

[33mcommit e03bc46a137ec03b50171e031b6b0f397956a20e[m
Author: Daisuke Nishimatsu <42202095+wep21@users.noreply.github.com>
Date:   Tue Aug 1 18:43:47 2023 +0900

    [Fix] modify download script for nuscenes (#2660)
    
    * fix: modify download script for nuscenes
    
    Signed-off-by: wep21 <daisuke.nishimatsu1021@gmail.com>
    
    * Update nuscenes_unzip.sh
    
    ---------
    
    Signed-off-by: wep21 <daisuke.nishimatsu1021@gmail.com>
    Co-authored-by: Xiang Xu <xuxiang0103@gmail.com>

[33mcommit 0f9dfa97a35ef87e16b700742d3c358d0ad15452[m[33m ([m[1;33mtag: [m[1;33mv1.2.0[m[33m)[m
Merge: e990105e d27e65aa
Author: ZwwWayne <wayne.zw@outlook.com>
Date:   Tue Jul 4 23:46:15 2023 +0800

    Merge branch 'main' into dev-1.x

[33mcommit e990105ed156c200f8bccfa43c268d47dcec59a4[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Tue Jul 4 23:35:30 2023 +0800

    [Fix] Fix indoor det visualization (#2625)
    
    * fix visual
    
    * add AttributeError
    
    * fix keep index bug
    
    * fix open3d version bug

[33mcommit fb5a3232b7ba26c010b9f984577162f02a056b00[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Tue Jul 4 23:27:46 2023 +0800

    Bump to v1.2.0 (#2630)
    
    * fix mmdet max version
    
    * add readme
    
    * fix changelog
    
    * fix keep index bug
    
    * fix open3d version bug

[33mcommit 12e2b7cf79d4ae52f459b082d67e5d42457e5584[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Sun Jun 25 14:35:18 2023 +0800

    [Enhance] Support LiDAR visualization (#2611)
    
    * need fix multi_modality
    
    * update multi modal
    
    * remove pcd
    
    * fix mix bug
    
    * fix nusence_mini
    
    * fix msehframe
    
    * fix flag exit
    
    * add space line

[33mcommit 2c1367304170bc17c176e551c2350eadc57a6f6b[m[33m ([m[1;31morigin/test-1.2[m[33m)[m
Author: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
Date:   Mon Jun 19 10:47:01 2023 +0800

    [Feature] Support new config type (#2608)
    
    * support new configs
    
    * support new configs
    
    * verify configs

[33mcommit aea26ac7cd544678db5bff157a658352449c29ac[m
Author: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
Date:   Mon Jun 19 10:44:05 2023 +0800

    Remove pointrcnn in benchmark training (#2610)

[33mcommit 8fb2cf6c3773977a3a1f473c43b065afc0181b48[m
Author: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
Date:   Fri Jun 16 10:06:26 2023 +0800

    [Feature] Support inference of DSVT in `projects` (#2606)
    
    * support inference
    
    * align inference precision
    
    * add readme
    
    * polish docs
    
    * polish docs

[33mcommit d27e65aac658e8e104dbbfd6a55119704010ea79[m
Author: JingweiZhang12 <zjw18@mails.tsinghua.edu.cn>
Date:   Thu Jun 15 11:55:31 2023 +0800

    [Fix] Fix typo

[33mcommit 456b7403cd9e7572cc47bb8bff6bcc17db882c38[m
Author: Xiang Xu <xuxiang0103@gmail.com>
Date:   Thu Jun 15 11:44:21 2023 +0800

    [Docs] Update README (#2599)
    
    * update readme
    
    * update readme
    
    * update readme

[33mcommit 148a0856ab47d4b4a92894838a4525adaddaaf0b[m
Author: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
Date:   Wed Jun 14 15:31:00 2023 +0800

    [Feature] Support downloading datasets from OpenDataLab using mim (#2593)
    
    * init commit
    
    * add dataset unzip scripts
    
    * polish docs
    
    * polish docs

[33mcommit f40d8d28f184fd932848af23f5fbcba75da28184[m
Author: ZhuLifa <zhulf0804@gmail.com>
Date:   Wed Jun 14 11:10:38 2023 +0800

    [Docs] Add 6 models in benchmark & model zoo in README.md (#2600)
    
    * [Docs]: Add BEVFusion in benchmark & model zoo in README
    
    * [Docs]: Add CenterFormer in benchmark & model zoo in README
    
    * [Docs]: Add TR3D in benchmark & model zoo in README
    
    * [Docs]: Add DETR3D and PETR in benchmark & model zoo in README
    
    * [Docs]: Add TPVFormer in benchmark & model zoo in README
    
    * [Docs]: Add 6 models in benchmark & model zoo in README_zh-CN
    
    * [Docs]: Add 6 models in model_zoo.md
    
    * [Style]: Stylize the code

[33mcommit e753ecb50977e0d76ebe29cc2c6fc726e1f744e2[m
Author: Maksim Kolodiazhnyi <col14m@gmail.com>
Date:   Tue Jun 6 11:21:56 2023 +0300

    [Enhance] Speed up loading text file for indoor datasets. (#2585)
    
    * Update indoor3d_util.py
    
    Changed data load function. Dataset parsing was accelerated from about 5 hours to 10 minutes.
    
    * Added pandas import
    
    * minor fix
    
    * fix markdown
    
    ---------
    
    Co-authored-by: JingweiZhang12 <zjw18@mails.tsinghua.edu.cn>

[33mcommit f5c558b2c959aabe3220acdde59246e7693cb0a2[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Mon Jun 5 14:21:51 2023 +0800

    [Docs] Update the link of minkunet (#2590)
    
    * update link
    
    * fix coord

[33mcommit 3fa0e8bbe63d4febb6b546df158e9746e76305ca[m
Author: JingweiZhang12 <zjw18@mails.tsinghua.edu.cn>
Date:   Wed May 31 17:11:53 2023 +0800

    [Fix] Remove unnecessary file fold

[33mcommit 5bb968c5a99123b2b8ef27fdd5df83e00f0b1869[m[33m ([m[1;33mtag: [m[1;33mv1.1.1[m[33m)[m
Merge: ca46c3b7 47e56e1d
Author: JingweiZhang12 <zjw18@mails.tsinghua.edu.cn>
Date:   Wed May 31 15:33:31 2023 +0800

    Merge branch 'main' of github.com:open-mmlab/mmdetection3d into dev-1.x

[33mcommit ca46c3b7463a0442a6e935761b959e6b11eef2f7[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Wed May 31 15:15:09 2023 +0800

    Bump to v1.1.1 (#2569)
    
    * add changelog
    
    * add readme
    
    * fix typo
    
    * Update README.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update README_zh-CN.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * Update docs/en/notes/changelog.md
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
    
    * fix review suggestion
    
    ---------
    
    Co-authored-by: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>

[33mcommit 68ea4b4d2de08b2fd195aed0a37471f3efa4ce7d[m
Author: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
Date:   Wed May 31 15:14:52 2023 +0800

    [Fix] Fix formula in the readthedocs (#2580)
    
    * fix latex in readthedocs
    
    * fix zh_cn latex
    
    * fix zh_cn latex
    
    * fix en latex

[33mcommit 583e9075530b302240c43bf5659349599c8a4164[m
Author: Jingwei Zhang <zjw18@mails.tsinghua.edu.cn>
Date:   Wed May 31 14:45:47 2023 +0800

    [Feature] Support training of BEVFusion (#2558)
    
    * support train on nus
    
    * refactor transfusion head
    
    * img branch optioinal
    
    * support nuscenes_mini in replace_ceph_backend
    
    * use replace_ceph
    
    * add only-lidar
    
    * use valid_flag in dataset filter
    
    * support lidar-only training 69
    
    * fix RTS
    
    * fix rotation in ImgAug3D
    
    * revert to original rotation in ImgAug3D
    
    * add LSSDepthTransform and parse_losses
    
    * fix LoadMultiSweeps
    
    * fix bug about points in-place operations
    
    * support amp and replace syncBN by BN
    
    * add amp config
    
    * set growth-interval in amp
    
    * Revert "fix LoadMultiSweeps"
    
    This reverts commit ab27ea1941f80c5d7fb3071005109f4794c9c9cb.
    
    * add float in cls loss
    
    * iter_based lr in fusion stage
    
    * rename config
    
    * use normalization query pos for stable training
    
    * remove unnecessary code & simplify config & train 5 epoch
    
    * smaller ete_min_ratio
    
    * polish code
    
    * fix UT
    
    * Revert "use normalization query pos for stable training"
    
    This reverts commit 30091188d4a665beaadcd5a9b49e04c9da47139e.
    
    * update readme
    
    * fix height offset

[33mcommit ed46b8c1da3656a32aa0e98ea5dd0d71cd0e6716[m
Author: Xiang Xu <xuxiang0103@gmail.com>
Date:   Tue May 30 16:58:37 2023 +0800

    [Enhance] Support modifying non_blocking parameters (#2567)

[33mcommit 56657c21ffb4b76d319979bbfd139bf625eb642b[m
Author: Sun Jiahao <72679458+sunjiahao1999@users.noreply.github.com>
Date:   Tue May 30 16:56:32 2023 +0800

    [Docs] Update readme of lidar segmentation methods  (#2559)
    
    * add readme
    
    * fix spvcnn memo