'use client';
import { useEffect, useRef } from 'react';
import * as echarts from 'echarts/core';
import { LineChart, BarChart, ScatterChart, HeatmapChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, VisualMapComponent, MarkLineComponent, MarkAreaComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
echarts.use([LineChart,BarChart,ScatterChart,HeatmapChart,GridComponent,TooltipComponent,LegendComponent,DataZoomComponent,VisualMapComponent,MarkLineComponent,MarkAreaComponent,CanvasRenderer]);
export default function Chart({option,height=280,onClick}:{option:any,height?:number,onClick?:(p:any)=>void}){
 const ref=useRef<HTMLDivElement>(null);
 useEffect(()=>{if(!ref.current)return;const chart=echarts.init(ref.current,undefined,{renderer:'canvas'});
 chart.setOption({animationDuration:250,backgroundColor:'transparent',color:['#83b6c9','#d2ab73','#8ea99b','#b78b8b'],textStyle:{fontFamily:'Arial, sans-serif',color:'#92a0aa',fontSize:11},grid:{left:55,right:24,top:28,bottom:42},tooltip:{trigger:'axis',backgroundColor:'#1b232b',borderColor:'#3b4651',textStyle:{color:'#e0e6eb',fontSize:12}},...option});
 if(onClick)chart.on('click',onClick);const resize=new ResizeObserver(()=>chart.resize());resize.observe(ref.current);return()=>{resize.disconnect();chart.dispose();};},[option,onClick]);
 return <div className="chart" ref={ref} style={{height}} role="img" aria-label="Interactive research chart"/>;
}
