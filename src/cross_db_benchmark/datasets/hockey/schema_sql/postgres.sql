

DROP TABLE IF EXISTS "AwardsCoaches";

CREATE TABLE "AwardsCoaches" (
  "coachID" varchar(13) DEFAULT NULL,
  "award" varchar(255) DEFAULT NULL,
  "year" integer DEFAULT NULL,
  "lgID" varchar(255) DEFAULT NULL,
  "note" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "AwardsMisc";

CREATE TABLE "AwardsMisc" (
  "name" varchar(255) ,
  "ID" varchar(255) DEFAULT NULL,
  "award" varchar(255) DEFAULT NULL,
  "year" integer DEFAULT NULL,
  "lgID" varchar(255) DEFAULT NULL,
  "note" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("name")
) ;

DROP TABLE IF EXISTS "AwardsPlayers";

CREATE TABLE "AwardsPlayers" (
  "playerID" varchar(12) ,
  "award" varchar(23) ,
  "year" integer ,
  "lgID" varchar(255) DEFAULT NULL,
  "note" varchar(255) DEFAULT NULL,
  "pos" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("playerID","award","year")
) ;

DROP TABLE IF EXISTS "Coaches";

CREATE TABLE "Coaches" (
  "coachID" varchar(13) ,
  "year" integer ,
  "tmID" varchar(6) ,
  "lgID" varchar(255) DEFAULT NULL,
  "stint" integer ,
  "notes" varchar(255) DEFAULT NULL,
  "g" integer DEFAULT NULL,
  "w" integer DEFAULT NULL,
  "l" integer DEFAULT NULL,
  "t" integer DEFAULT NULL,
  "postg" varchar(255) DEFAULT NULL,
  "postw" varchar(255) DEFAULT NULL,
  "postl" varchar(255) DEFAULT NULL,
  "postt" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("coachID","year","tmID","stint")
) ;

DROP TABLE IF EXISTS "CombinedShutouts";

CREATE TABLE "CombinedShutouts" (
  "year" integer DEFAULT NULL,
  "month" integer DEFAULT NULL,
  "date" integer DEFAULT NULL,
  "tmID" varchar(255) DEFAULT NULL,
  "oppID" varchar(255) DEFAULT NULL,
  "R/P" varchar(255) DEFAULT NULL,
  "IDgoalie1" varchar(12) DEFAULT NULL,
  "IDgoalie2" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "Goalies";

CREATE TABLE "Goalies" (
  "playerID" varchar(12) ,
  "year" integer ,
  "stint" integer ,
  "tmID" varchar(6) DEFAULT NULL,
  "lgID" varchar(255) DEFAULT NULL,
  "GP" varchar(255) DEFAULT NULL,
  "Min" varchar(255) DEFAULT NULL,
  "W" varchar(255) DEFAULT NULL,
  "L" varchar(255) DEFAULT NULL,
  "T/OL" varchar(255) DEFAULT NULL,
  "ENG" varchar(255) DEFAULT NULL,
  "SHO" varchar(255) DEFAULT NULL,
  "GA" varchar(255) DEFAULT NULL,
  "SA" varchar(255) DEFAULT NULL,
  "PostGP" varchar(255) DEFAULT NULL,
  "PostMin" varchar(255) DEFAULT NULL,
  "PostW" varchar(255) DEFAULT NULL,
  "PostL" varchar(255) DEFAULT NULL,
  "PostT" varchar(255) DEFAULT NULL,
  "PostENG" varchar(255) DEFAULT NULL,
  "PostSHO" varchar(255) DEFAULT NULL,
  "PostGA" varchar(255) DEFAULT NULL,
  "PostSA" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("playerID","year","stint")
) ;

DROP TABLE IF EXISTS "GoaliesSC";

CREATE TABLE "GoaliesSC" (
  "playerID" varchar(12) ,
  "year" integer ,
  "tmID" varchar(255) DEFAULT NULL,
  "lgID" varchar(255) DEFAULT NULL,
  "GP" integer DEFAULT NULL,
  "Min" integer DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "T" integer DEFAULT NULL,
  "SHO" integer DEFAULT NULL,
  "GA" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","year")
) ;

DROP TABLE IF EXISTS "GoaliesShootout";

CREATE TABLE "GoaliesShootout" (
  "playerID" varchar(12) DEFAULT NULL,
  "year" integer DEFAULT NULL,
  "stint" integer DEFAULT NULL,
  "tmID" varchar(255) DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "SA" integer DEFAULT NULL,
  "GA" integer DEFAULT NULL
) ;

DROP TABLE IF EXISTS "HOF";

CREATE TABLE "HOF" (
  "year" integer DEFAULT NULL,
  "hofID" varchar(255) ,
  "name" varchar(255) DEFAULT NULL,
  "category" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("hofID")
) ;

DROP TABLE IF EXISTS "Master";

CREATE TABLE "Master" (
  "playerID" varchar(12) DEFAULT NULL,
  "coachID" varchar(255) DEFAULT NULL,
  "hofID" varchar(255) DEFAULT NULL,
  "firstName" varchar(255) DEFAULT NULL,
  "lastName" varchar(255) ,
  "nameNote" varchar(255) DEFAULT NULL,
  "nameGiven" varchar(255) DEFAULT NULL,
  "nameNick" varchar(255) DEFAULT NULL,
  "height" varchar(255) DEFAULT NULL,
  "weight" varchar(255) DEFAULT NULL,
  "shootCatch" varchar(255) DEFAULT NULL,
  "legendsID" varchar(255) DEFAULT NULL,
  "ihdbID" varchar(255) DEFAULT NULL,
  "hrefID" varchar(255) DEFAULT NULL,
  "firstNHL" varchar(255) DEFAULT NULL,
  "lastNHL" varchar(255) DEFAULT NULL,
  "firstWHA" varchar(255) DEFAULT NULL,
  "lastWHA" varchar(255) DEFAULT NULL,
  "pos" varchar(255) DEFAULT NULL,
  "birthYear" varchar(255) DEFAULT NULL,
  "birthMon" varchar(255) DEFAULT NULL,
  "birthDay" varchar(255) DEFAULT NULL,
  "birthCountry" varchar(255) DEFAULT NULL,
  "birthState" varchar(255) DEFAULT NULL,
  "birthCity" varchar(255) DEFAULT NULL,
  "deathYear" varchar(255) DEFAULT NULL,
  "deathMon" varchar(255) DEFAULT NULL,
  "deathDay" varchar(255) DEFAULT NULL,
  "deathCountry" varchar(255) DEFAULT NULL,
  "deathState" varchar(255) DEFAULT NULL,
  "deathCity" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "Scoring";

CREATE TABLE "Scoring" (
  "playerID" varchar(12) DEFAULT NULL,
  "year" integer DEFAULT NULL,
  "stint" integer DEFAULT NULL,
  "tmID" varchar(255) DEFAULT NULL,
  "lgID" varchar(255) DEFAULT NULL,
  "pos" varchar(255) DEFAULT NULL,
  "GP" integer DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "A" integer DEFAULT NULL,
  "Pts" integer DEFAULT NULL,
  "PIM" integer DEFAULT NULL,
  "+/-" varchar(255) DEFAULT NULL,
  "PPG" varchar(255) DEFAULT NULL,
  "PPA" varchar(255) DEFAULT NULL,
  "SHG" varchar(255) DEFAULT NULL,
  "SHA" varchar(255) DEFAULT NULL,
  "GWG" varchar(255) DEFAULT NULL,
  "GTG" varchar(255) DEFAULT NULL,
  "SOG" varchar(255) DEFAULT NULL,
  "PostGP" varchar(255) DEFAULT NULL,
  "PostG" varchar(255) DEFAULT NULL,
  "PostA" varchar(255) DEFAULT NULL,
  "PostPts" varchar(255) DEFAULT NULL,
  "PostPIM" varchar(255) DEFAULT NULL,
  "Post+/-" varchar(255) DEFAULT NULL,
  "PostPPG" varchar(255) DEFAULT NULL,
  "PostPPA" varchar(255) DEFAULT NULL,
  "PostSHG" varchar(255) DEFAULT NULL,
  "PostSHA" varchar(255) DEFAULT NULL,
  "PostGWG" varchar(255) DEFAULT NULL,
  "PostSOG" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "ScoringSC";

CREATE TABLE "ScoringSC" (
  "playerID" varchar(12) DEFAULT NULL,
  "year" integer DEFAULT NULL,
  "tmID" varchar(255) DEFAULT NULL,
  "lgID" varchar(255) DEFAULT NULL,
  "pos" varchar(255) DEFAULT NULL,
  "GP" integer DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "A" integer DEFAULT NULL,
  "Pts" integer DEFAULT NULL,
  "PIM" integer DEFAULT NULL
) ;

DROP TABLE IF EXISTS "ScoringShootout";

CREATE TABLE "ScoringShootout" (
  "playerID" varchar(12) DEFAULT NULL,
  "year" integer DEFAULT NULL,
  "stint" integer DEFAULT NULL,
  "tmID" varchar(255) DEFAULT NULL,
  "S" integer DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "GDG" integer DEFAULT NULL
) ;

DROP TABLE IF EXISTS "ScoringSup";

CREATE TABLE "ScoringSup" (
  "playerID" varchar(12) DEFAULT NULL,
  "year" integer DEFAULT NULL,
  "PPA" varchar(255) DEFAULT NULL,
  "SHA" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "SeriesPost";

CREATE TABLE "SeriesPost" (
  "year" integer DEFAULT NULL,
  "round" varchar(255) DEFAULT NULL,
  "series" varchar(255) DEFAULT NULL,
  "tmIDWinner" varchar(6) DEFAULT NULL,
  "lgIDWinner" varchar(255) DEFAULT NULL,
  "tmIDLoser" varchar(255) DEFAULT NULL,
  "lgIDLoser" varchar(255) DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "T" integer DEFAULT NULL,
  "GoalsWinner" integer DEFAULT NULL,
  "GoalsLoser" integer DEFAULT NULL,
  "note" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "TeamSplits";

CREATE TABLE "TeamSplits" (
  "year" integer ,
  "lgID" varchar(255) DEFAULT NULL,
  "tmID" varchar(6) ,
  "hW" integer DEFAULT NULL,
  "hL" integer DEFAULT NULL,
  "hT" integer DEFAULT NULL,
  "hOTL" varchar(255) DEFAULT NULL,
  "rW" integer DEFAULT NULL,
  "rL" integer DEFAULT NULL,
  "rT" integer DEFAULT NULL,
  "rOTL" varchar(255) DEFAULT NULL,
  "SepW" varchar(255) DEFAULT NULL,
  "SepL" varchar(255) DEFAULT NULL,
  "SepT" varchar(255) DEFAULT NULL,
  "SepOL" varchar(255) DEFAULT NULL,
  "OctW" varchar(255) DEFAULT NULL,
  "OctL" varchar(255) DEFAULT NULL,
  "OctT" varchar(255) DEFAULT NULL,
  "OctOL" varchar(255) DEFAULT NULL,
  "NovW" varchar(255) DEFAULT NULL,
  "NovL" varchar(255) DEFAULT NULL,
  "NovT" varchar(255) DEFAULT NULL,
  "NovOL" varchar(255) DEFAULT NULL,
  "DecW" varchar(255) DEFAULT NULL,
  "DecL" varchar(255) DEFAULT NULL,
  "DecT" varchar(255) DEFAULT NULL,
  "DecOL" varchar(255) DEFAULT NULL,
  "JanW" integer DEFAULT NULL,
  "JanL" integer DEFAULT NULL,
  "JanT" integer DEFAULT NULL,
  "JanOL" varchar(255) DEFAULT NULL,
  "FebW" integer DEFAULT NULL,
  "FebL" integer DEFAULT NULL,
  "FebT" integer DEFAULT NULL,
  "FebOL" varchar(255) DEFAULT NULL,
  "MarW" varchar(255) DEFAULT NULL,
  "MarL" varchar(255) DEFAULT NULL,
  "MarT" varchar(255) DEFAULT NULL,
  "MarOL" varchar(255) DEFAULT NULL,
  "AprW" varchar(255) DEFAULT NULL,
  "AprL" varchar(255) DEFAULT NULL,
  "AprT" varchar(255) DEFAULT NULL,
  "AprOL" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("year","tmID")
) ;

DROP TABLE IF EXISTS "TeamVsTeam";

CREATE TABLE "TeamVsTeam" (
  "year" integer ,
  "lgID" varchar(255) DEFAULT NULL,
  "tmID" varchar(6) ,
  "oppID" varchar(6) ,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "T" integer DEFAULT NULL,
  "OTL" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("year","tmID","oppID")
) ;

DROP TABLE IF EXISTS "Teams";

CREATE TABLE "Teams" (
  "year" integer ,
  "lgID" varchar(255) DEFAULT NULL,
  "tmID" varchar(6) ,
  "franchID" varchar(255) DEFAULT NULL,
  "confID" varchar(255) DEFAULT NULL,
  "divID" varchar(255) DEFAULT NULL,
  "rank" integer DEFAULT NULL,
  "playoff" varchar(255) DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "T" integer DEFAULT NULL,
  "OTL" varchar(255) DEFAULT NULL,
  "Pts" integer DEFAULT NULL,
  "SoW" varchar(255) DEFAULT NULL,
  "SoL" varchar(255) DEFAULT NULL,
  "GF" integer DEFAULT NULL,
  "GA" integer DEFAULT NULL,
  "name" varchar(255) DEFAULT NULL,
  "PIM" varchar(255) DEFAULT NULL,
  "BenchMinor" varchar(255) DEFAULT NULL,
  "PPG" varchar(255) DEFAULT NULL,
  "PPC" varchar(255) DEFAULT NULL,
  "SHA" varchar(255) DEFAULT NULL,
  "PKG" varchar(255) DEFAULT NULL,
  "PKC" varchar(255) DEFAULT NULL,
  "SHF" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("year","tmID")
) ;

DROP TABLE IF EXISTS "TeamsHalf";

CREATE TABLE "TeamsHalf" (
  "year" integer ,
  "lgID" varchar(255) DEFAULT NULL,
  "tmID" varchar(6) ,
  "half" integer ,
  "rank" integer DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "T" integer DEFAULT NULL,
  "GF" integer DEFAULT NULL,
  "GA" integer DEFAULT NULL,
  PRIMARY KEY ("year","tmID","half")
) ;

DROP TABLE IF EXISTS "TeamsPost";

CREATE TABLE "TeamsPost" (
  "year" integer ,
  "lgID" varchar(255) DEFAULT NULL,
  "tmID" varchar(6) ,
  "G" integer DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "T" integer DEFAULT NULL,
  "GF" integer DEFAULT NULL,
  "GA" integer DEFAULT NULL,
  "PIM" varchar(255) DEFAULT NULL,
  "BenchMinor" varchar(255) DEFAULT NULL,
  "PPG" varchar(255) DEFAULT NULL,
  "PPC" varchar(255) DEFAULT NULL,
  "SHA" varchar(255) DEFAULT NULL,
  "PKG" varchar(255) DEFAULT NULL,
  "PKC" varchar(255) DEFAULT NULL,
  "SHF" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("year","tmID")
) ;

DROP TABLE IF EXISTS "TeamsSC";

CREATE TABLE "TeamsSC" (
  "year" integer ,
  "lgID" varchar(255) DEFAULT NULL,
  "tmID" varchar(6) ,
  "G" integer DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "T" integer DEFAULT NULL,
  "GF" integer DEFAULT NULL,
  "GA" integer DEFAULT NULL,
  "PIM" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("year","tmID")
) ;

DROP TABLE IF EXISTS "abbrev";

CREATE TABLE "abbrev" (
  "Type" varchar(255) ,
  "Code" varchar(255) ,
  "Fullname" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Type","Code")
) ;

