var annplotRad = 17.95;
var subplotRad = 7.32;
var buf = 0.00028;
var centroid1 = [-120.18772870302200, 39.09959321926681];
var centroid2 = [-120.18866747617722, 39.11031853091724];

function generateFIA(centroid, bufferDist) {
        var deltayNorth = 0.00028;
        var deltaySouth = 0.0002075;
        var deltax = 0.00028;
        var out = /* color: #0b4a8b */ee.Feature(
                    ee.Geometry.MultiPoint(
                                [[centroid[0], centroid[1]],
                                             [centroid[0], centroid[1]+deltayNorth],
                                             [centroid[0]-deltax, centroid[1]-deltaySouth],
                                             [centroid[0]+deltax, centroid[1]-deltaySouth]]))
             .buffer(bufferDist);
        return out;              
}

function generateBoundary(centroid, bufferDist){
      if (bufferDist === undefined) {
                  var b = 0.0006;
                  bufferDist = [b+0.000175,b+0.000075,b];
              }
      var bounds = ee.Geometry.Polygon([[[centroid[0]-bufferDist[0], centroid[1]+bufferDist[1]],
            [centroid[0]+bufferDist[0], centroid[1]+bufferDist[1]],
            [centroid[0]+bufferDist[0], centroid[1]-bufferDist[2]],
            [centroid[0]-bufferDist[0], centroid[1]-bufferDist[2]],
            [centroid[0]-bufferDist[0], centroid[1]+bufferDist[1]]]]);
      return bounds;
}

var fiaSubplots = generateFIA(centroid1, subplotRad);
var fiaSubplots2 = generateFIA(centroid2, subplotRad);

var bounds = generateBoundary(centroid1)

// DATE RANGE
var start = ee.Date('2009-01-01');
var finish = ee.Date('2011-12-31');

// DEFINE LANDSAT
var LT5 = ee.ImageCollection('LANDSAT/LT5_L1T_TOA');
var maskClouds = function(image) {
  var scored = ee.Algorithms.Landsat.simpleCloudScore(image);
    return image.updateMask(scored.select(['cloud']).lt(20));
    };
    var filteredLT5 = LT5.filterBounds(bounds).
    filterDate(start, finish).map(maskClouds);
    var outLT5 = filteredLT5.map(function(img){
      return(img.clip(bounds))
      })

    var vizParams = {
        bands: ['B3', 'B2', 'B1'],
        min: 0,
        max: 0.5,
        gamma: [0.95, 1.1, 1],
        opacity: 0.5
    };

    var demVizParams = {
        bands: ["elevation"],
        gamma: 1,
        max: 3712,
        min: -82,
        opacity: 1
    };
    var outImg = outLT5.map(function(image){
        return image.visualize(vizParams)
        })
// DEFINE NAIP
var NAIP = ee.ImageCollection('USDA/NAIP/DOQQ');
var filteredNAIP = NAIP.filterBounds(bounds).
    filterDate('2013-01-01', '2017-12-31')
var outNAIP = filteredNAIP.map(function(img){
    return(img.clip(bounds))
    })
// DEFINE ROI
var ROI = ee.FeatureCollection('ft:1EJN9oLRyAVKAsx-mXUT2Gv59PWEIj1yWRBroPUWH');
var empty = ee.Image().byte();
// Paint all the polygon edges with the same number and width, display.
var ROIoutline = empty.paint({
    featureCollection: ROI,
    color: 1,
    width: 3
    });
// Paint all the polygon edges with the same number and width, display.
Map.addLayer(dem.clip(bounds), demVizParams, 'DEM');
Map.addLayer(ROIoutline, {palette: 'FF0000'}, 'ROI');
Map.addLayer(outNAIP, {bands: ['R','G','B']}, 'NAIP');
Map.addLayer(outLT5, vizParams, 'LANDSAT');
Map.addLayer(fiaSubplots, {opacity:0.6}, 'FIA Plot Ex 1');//look up paint in documents as images
Map.addLayer(fiaSubplots2, {opacity:0.6}, 'FIA Plot Ex 2');//look up paint in documents as images
Map.centerObject(outNAIP, 13);
//

/*
Export.video.toCloudStorage({'collection':outImg, 'description':'test_run_gif', 
'bucket':'wood-supply', 'fileNamePrefix':'exampleExport', 
'dimensions':400, 'framesPerSecond':5})

outLT5.visualize(bands, gain, bias, min, max, gamma, opacity, palette, forceRgbOutput)

Export.video.toCloudStorage(collection, description, bucket, 
fileNamePrefix, framesPerSecond, dimensions, region, scale, 
maxFrames)
*/
