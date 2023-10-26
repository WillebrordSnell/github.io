import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/keyan/": [

    "videoUnderstanding/videoUnderstanding",
    "videoRepresentation/videoRepresentation",
    "videoDialog/videoDialog",
    "contrastiveLearning/contrastiveLearning",
    "multiModal/multiModal",

  ],

  "/book/": [

    "202309" ,
    "202310" ,
    "maoxuan" ,
  ],

  "/train/": [

    "DDP/DDP" ,
    "AVSD/AVSD" ,
    "trick/trick" ,
  ],

  "/Tools/Git/": "structure",
  "/Tools/Git/*": "structure",
  "/Tools/MarkDown/": "structure",
  "/Tools/Resource/": "structure",

});
