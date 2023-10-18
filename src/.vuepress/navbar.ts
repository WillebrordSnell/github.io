import { navbar } from "vuepress-theme-hope";

export default navbar([

  // 
  { 
    text: "码头",
    prefix: "/keyan/",
    children: [
        { text: "视频理解", link: "videoUnderstanding/_videoUnderstanding" },
        { text: "视频表征", link: "videoRepresentation/_videoRepresentation" },
        { text: "对比学习", link: "contrastiveLearning/_contrastiveLearning" },
        { text: "多模态",   link: "multiModal/_multiModal" },  


    ]
  },
  {
    text: "炉",
    prefix: "/train/",
    children: [
      { text: "单机多卡DDP", link: "DDP/_DDP" },
      { text: "AVSD", link: "AVSD/_AVSD" },
      { text: "奇淫技巧", link: "trick/_trick" },
    ]
  },

  {
    text: "道心",
    prefix: "/book/",
    children: [
      { text: "2023年9月",  link: "202309" },
      { text: "2023年10月", link: "202310" },
      { text: "毛泽东选集", link: "maoxuan" },
    ],
  },


  // {
  //   text: "金樽",
  //   prefix: "/life/",
  //   children: [
  //     {
  //       text: "游戏",

  //       prefix: "game/",
  //       children: [
  //         { text: "死亡搁浅", link: "DeathStranding" },
  //         { text: "血缘诅咒", link: "Bloodborne" },
  //       ],
  //     },


  //   ],
  // },


  /* {
    text: "V2 文档",
    icon: "book",
    link: "https://theme-hope.vuejs.press/zh/",
  },*/
]);
