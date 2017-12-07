CDRC 2011 Population Weighted Centroids - GB


+ Abstract
The CDRC 2011 Population Weighted Centroids (LSOA/Data Zone) - GB data pack is integral of data from multiple sources which renders population weighted centroids for each LSOA or Data Zone in Great Britain and correspondent population and households. 


+ Contents
	 - readme.txt: Information about the CDRC Geodata pack
	 - metadata.xml: Metadata
	 - fields_lookup.csv: Lookup table for the datasets field names
	 - tables: Folder containing the csv files
	 - shapefiles: Folder containing the shapefiles

+ Population Weighted Centroids (descriptions excerted from source data packs)

Scottish centroids for data zone boundaries have been created by the Scottish Government's Geographic Information Science and Analysis Team (GI-SAT) in order to provide a way to link data zones to other (higher level) geographic boundaries used by Scottish Neighbourhood Statistics (SNS) and the wider public sector. Data zone centroids do not represent the geometric centre of the feature, but rather the population weighted centre. A detailed methodology to the process used in 2001 can be found at: http://www.scotland.gov.uk/Resource/Doc/933/0082884.docWhile the method used to create 2011 Data Zone Centroids remains broadly the same, a small change was implemented following consultation with Local Authorities regarding 2011 Data Zone boundaries.  This change was to use the median of locations of output area centroids contained within a data zone, as opposed to the mean centre.  The median is a measure of central tendency and, broadly speaking, the median can be thought of as the 'middle' value. While the mean is calculated by summing all the values together and then dividing by the number of observations, the median is calculated by putting the observations in order, from lowest to highest, and then taking the value in the middle. (Or calculating the mean of the two middle values if there are an even number of observations.).  The key advantage of using the median is that it is not as heavily influenced by extreme values as the mean. If a Data Zone has a highly skewed population distribution, for example a large rural data zone containing a small town in one corner, then the mean can be heavily influenced by the small number of people who live far away from the population centre and the mean will likely fall outside of the town. The median is considered to be a more robust measure of central tendency and is less likely to be influenced by values far away from what would be considered to be the population centre of the Data Zone.The process for creating 2011 Data Zone Centroids was automated using ESRI ArcGIS, but the general method is as follows.  The median easting and northing coordinate pair for all 2011 Census Output Areas within the Data Zone is calculated, giving a notional centroid of the Data Zone. Since data zones can be complex shapes, a second step carried out to ensure that the median falls within the data zone boundary. The distance from each of the Census Output Area centroids to the notional (or median) centroid is calculated using Pythagoras' Theorem. The Census Output Area coordinate pair with the shortest distance to the median was then chosen to represent the centroid of the Data Zone.Each 2011 Data Zone has been given a new unique code, following the Scottish Government’s standard naming and coding convention. The Data Zone 2011 codes range from S01006506 to S01013481 (the previous 2001 codes ranged from S01000001 to S01006505). Most data zones have been named according to the Intermediate Zone in which they reside, following the format ‘Intermdiate Zone Name – 01’, ‘Intermediate Zone Name – 02’, etc. Some Councils chose to provide an individual name to each data zone (e.g. Fife). Census 2011 total, resident and household counts have been summed for each data zone and included in the attribute table.


Each instance of 2011 LSOA in England and Welsh has a population weighted centroid, that reflects the spatial distribution of the 2011 Census population in each instance of those geographies, referenced to a single summary point on the ground. Each population weighted centroid was calculated using a median centroid algorithm, which is less influenced by outliers than a mean centroid algorithm.  The median algorithm used was the Median Center (sic) function in ArcGIS 10, run against the  coordinates and the populations of each household in each OA, LSOA and MSOA.  Where the calculated centroid fell outside the boundary of the area being calculated, or within two metres of the area boundary, it was moved to the nearest location at least two metres inside the area boundary.   More information about how the Median Center algorithm works can be found in the link below.
The centroids were created using Full Resolution Extent of the Realm boundaries.

+ Citation and Copyright

The following attribution statements must be used to acknowledge copyright and source in use of these datasets:
               Contains National Statistics data © Crown copyright and database right 2015;
               Contains Sottish Government data © Crown copyright and database right 2015;
               Contains Ordnance Survey data © Crown copyright and database right 2015;
               Data provided by the ESRC Consumer Data Research Centre.


+ Funding

Funded by: Economic and Social Research Council ES/L011840/1
