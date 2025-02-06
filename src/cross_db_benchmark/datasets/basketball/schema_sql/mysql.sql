-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: Basketball_men
-- ------------------------------------------------------
-- Server version	5.5.5-10.3.15-MariaDB-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `awards_coaches`
--

DROP TABLE IF EXISTS `awards_coaches`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `awards_coaches` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `year` int(11) DEFAULT NULL,
  `coachID` varchar(255) DEFAULT NULL,
  `award` varchar(255) DEFAULT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `note` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `coachID` (`coachID`),
  KEY `year` (`year`,`coachID`),
  KEY `year_2` (`year`,`coachID`),
  KEY `year_3` (`year`,`coachID`,`lgID`),
  KEY `coachID_2` (`coachID`,`year`),
  KEY `coachID_3` (`coachID`,`year`),
  CONSTRAINT `awards_coaches_ibfk_1` FOREIGN KEY (`coachID`, `year`) REFERENCES `coaches` (`coachID`, `year`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=62 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `awards_players`
--

DROP TABLE IF EXISTS `awards_players`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `awards_players` (
  `playerID` varchar(255) NOT NULL,
  `award` varchar(255) NOT NULL,
  `year` int(11) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `note` varchar(255) DEFAULT NULL,
  `pos` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`year`,`award`),
  KEY `playerID` (`playerID`),
  CONSTRAINT `awards_players_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `coaches`
--

DROP TABLE IF EXISTS `coaches`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `coaches` (
  `coachID` varchar(255) NOT NULL,
  `year` int(11) NOT NULL,
  `tmID` varchar(255) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `stint` int(11) NOT NULL,
  `won` int(11) DEFAULT NULL,
  `lost` int(11) DEFAULT NULL,
  `post_wins` int(11) DEFAULT NULL,
  `post_losses` int(11) DEFAULT NULL,
  PRIMARY KEY (`coachID`,`year`,`tmID`,`stint`),
  KEY `coachID` (`coachID`),
  KEY `year` (`year`,`coachID`),
  KEY `year_2` (`year`,`coachID`,`lgID`),
  KEY `year_3` (`year`,`tmID`),
  KEY `year_4` (`year`,`tmID`),
  KEY `tmID` (`tmID`,`year`),
  KEY `tmID_2` (`tmID`,`year`),
  CONSTRAINT `coaches_ibfk_1` FOREIGN KEY (`tmID`, `year`) REFERENCES `teams` (`tmID`, `year`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `draft`
--

DROP TABLE IF EXISTS `draft`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `draft` (
  `id` int(11) NOT NULL DEFAULT 0,
  `draftYear` int(11) DEFAULT NULL,
  `draftRound` int(11) DEFAULT NULL,
  `draftSelection` int(11) DEFAULT NULL,
  `draftOverall` int(11) DEFAULT NULL,
  `tmID` varchar(255) DEFAULT NULL,
  `firstName` varchar(255) DEFAULT NULL,
  `lastName` varchar(255) DEFAULT NULL,
  `suffixName` varchar(255) DEFAULT NULL,
  `playerID` varchar(255) DEFAULT NULL,
  `draftFrom` varchar(255) DEFAULT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `tmID` (`tmID`,`draftYear`),
  CONSTRAINT `draft_ibfk_1` FOREIGN KEY (`tmID`, `draftYear`) REFERENCES `teams` (`tmID`, `year`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `player_allstar`
--

DROP TABLE IF EXISTS `player_allstar`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `player_allstar` (
  `playerID` varchar(255) NOT NULL,
  `last_name` varchar(255) DEFAULT NULL,
  `first_name` varchar(255) DEFAULT NULL,
  `season_id` int(11) NOT NULL,
  `conference` varchar(255) DEFAULT NULL,
  `league_id` varchar(255) DEFAULT NULL,
  `games_played` int(11) DEFAULT NULL,
  `minutes` int(11) DEFAULT NULL,
  `points` int(11) DEFAULT NULL,
  `o_rebounds` int(11) DEFAULT NULL,
  `d_rebounds` int(11) DEFAULT NULL,
  `rebounds` int(11) DEFAULT NULL,
  `assists` int(11) DEFAULT NULL,
  `steals` int(11) DEFAULT NULL,
  `blocks` int(11) DEFAULT NULL,
  `turnovers` int(11) DEFAULT NULL,
  `personal_fouls` int(11) DEFAULT NULL,
  `fg_attempted` int(11) DEFAULT NULL,
  `fg_made` int(11) DEFAULT NULL,
  `ft_attempted` int(11) DEFAULT NULL,
  `ft_made` int(11) DEFAULT NULL,
  `three_attempted` int(11) DEFAULT NULL,
  `three_made` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`season_id`),
  KEY `player_id` (`playerID`),
  CONSTRAINT `player_allstar_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `players`
--

DROP TABLE IF EXISTS `players`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `players` (
  `playerID` varchar(255) NOT NULL,
  `useFirst` varchar(255) DEFAULT NULL,
  `firstName` varchar(255) DEFAULT NULL,
  `middleName` varchar(255) DEFAULT NULL,
  `lastName` varchar(255) DEFAULT NULL,
  `nameGiven` varchar(255) DEFAULT NULL,
  `fullGivenName` varchar(255) DEFAULT NULL,
  `nameSuffix` varchar(255) DEFAULT NULL,
  `nameNick` varchar(255) DEFAULT NULL,
  `pos` varchar(255) DEFAULT NULL,
  `firstseason` int(11) DEFAULT NULL,
  `lastseason` int(11) DEFAULT NULL,
  `height` float DEFAULT NULL,
  `weight` int(11) DEFAULT NULL,
  `college` varchar(255) DEFAULT NULL,
  `collegeOther` varchar(255) DEFAULT NULL,
  `birthDate` date DEFAULT NULL,
  `birthCity` varchar(255) DEFAULT NULL,
  `birthState` varchar(255) DEFAULT NULL,
  `birthCountry` varchar(255) DEFAULT NULL,
  `highSchool` varchar(255) DEFAULT NULL,
  `hsCity` varchar(255) DEFAULT NULL,
  `hsState` varchar(255) DEFAULT NULL,
  `hsCountry` varchar(255) DEFAULT NULL,
  `deathDate` date DEFAULT NULL,
  `race` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`playerID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `players_teams`
--

DROP TABLE IF EXISTS `players_teams`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `players_teams` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `playerID` varchar(255) NOT NULL,
  `year` int(11) DEFAULT NULL,
  `stint` int(11) DEFAULT NULL,
  `tmID` varchar(255) DEFAULT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `GP` int(11) DEFAULT NULL,
  `GS` int(11) DEFAULT NULL,
  `minutes` int(11) DEFAULT NULL,
  `points` int(11) DEFAULT NULL,
  `oRebounds` int(11) DEFAULT NULL,
  `dRebounds` int(11) DEFAULT NULL,
  `rebounds` int(11) DEFAULT NULL,
  `assists` int(11) DEFAULT NULL,
  `steals` int(11) DEFAULT NULL,
  `blocks` int(11) DEFAULT NULL,
  `turnovers` int(11) DEFAULT NULL,
  `PF` int(11) DEFAULT NULL,
  `fgAttempted` int(11) DEFAULT NULL,
  `fgMade` int(11) DEFAULT NULL,
  `ftAttempted` int(11) DEFAULT NULL,
  `ftMade` int(11) DEFAULT NULL,
  `threeAttempted` int(11) DEFAULT NULL,
  `threeMade` int(11) DEFAULT NULL,
  `PostGP` int(11) DEFAULT NULL,
  `PostGS` int(11) DEFAULT NULL,
  `PostMinutes` int(11) DEFAULT NULL,
  `PostPoints` int(11) DEFAULT NULL,
  `PostoRebounds` int(11) DEFAULT NULL,
  `PostdRebounds` int(11) DEFAULT NULL,
  `PostRebounds` int(11) DEFAULT NULL,
  `PostAssists` int(11) DEFAULT NULL,
  `PostSteals` int(11) DEFAULT NULL,
  `PostBlocks` int(11) DEFAULT NULL,
  `PostTurnovers` int(11) DEFAULT NULL,
  `PostPF` int(11) DEFAULT NULL,
  `PostfgAttempted` int(11) DEFAULT NULL,
  `PostfgMade` int(11) DEFAULT NULL,
  `PostftAttempted` int(11) DEFAULT NULL,
  `PostftMade` int(11) DEFAULT NULL,
  `PostthreeAttempted` int(11) DEFAULT NULL,
  `PostthreeMade` int(11) DEFAULT NULL,
  `note` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `playerID` (`playerID`),
  KEY `year` (`year`,`tmID`),
  KEY `tmID` (`tmID`,`year`),
  KEY `tmID_2` (`tmID`,`year`),
  KEY `tmID_3` (`tmID`,`year`),
  KEY `fgAttempted` (`fgAttempted`),
  KEY `fgAttempted_2` (`fgAttempted`),
  CONSTRAINT `players_teams_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `players_teams_ibfk_2` FOREIGN KEY (`tmID`, `year`) REFERENCES `teams` (`tmID`, `year`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=23752 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `series_post`
--

DROP TABLE IF EXISTS `series_post`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `series_post` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `year` int(11) DEFAULT NULL,
  `round` varchar(255) DEFAULT NULL,
  `series` varchar(255) DEFAULT NULL,
  `tmIDWinner` varchar(255) DEFAULT NULL,
  `lgIDWinner` varchar(255) DEFAULT NULL,
  `tmIDLoser` varchar(255) DEFAULT NULL,
  `lgIDLoser` varchar(255) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `tmIDWinner` (`tmIDWinner`,`year`) USING BTREE,
  KEY `tmIDLoser` (`tmIDLoser`,`year`) USING BTREE,
  CONSTRAINT `series_post_ibfk_1` FOREIGN KEY (`tmIDWinner`, `year`) REFERENCES `teams` (`tmID`, `year`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `series_post_ibfk_2` FOREIGN KEY (`tmIDLoser`, `year`) REFERENCES `teams` (`tmID`, `year`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=776 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `teams`
--

DROP TABLE IF EXISTS `teams`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `teams` (
  `year` int(11) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `tmID` varchar(255) NOT NULL,
  `franchID` varchar(255) DEFAULT NULL,
  `confID` varchar(255) DEFAULT NULL,
  `divID` varchar(255) DEFAULT NULL,
  `rank` int(11) DEFAULT NULL,
  `confRank` int(11) DEFAULT NULL,
  `playoff` varchar(255) DEFAULT NULL,
  `name` varchar(255) DEFAULT NULL,
  `o_fgm` int(11) DEFAULT NULL,
  `o_fga` int(11) DEFAULT NULL,
  `o_ftm` int(11) DEFAULT NULL,
  `o_fta` int(11) DEFAULT NULL,
  `o_3pm` int(11) DEFAULT NULL,
  `o_3pa` int(11) DEFAULT NULL,
  `o_oreb` int(11) DEFAULT NULL,
  `o_dreb` int(11) DEFAULT NULL,
  `o_reb` int(11) DEFAULT NULL,
  `o_asts` int(11) DEFAULT NULL,
  `o_pf` int(11) DEFAULT NULL,
  `o_stl` int(11) DEFAULT NULL,
  `o_to` int(11) DEFAULT NULL,
  `o_blk` int(11) DEFAULT NULL,
  `o_pts` int(11) DEFAULT NULL,
  `d_fgm` int(11) DEFAULT NULL,
  `d_fga` int(11) DEFAULT NULL,
  `d_ftm` int(11) DEFAULT NULL,
  `d_fta` int(11) DEFAULT NULL,
  `d_3pm` int(11) DEFAULT NULL,
  `d_3pa` int(11) DEFAULT NULL,
  `d_oreb` int(11) DEFAULT NULL,
  `d_dreb` int(11) DEFAULT NULL,
  `d_reb` int(11) DEFAULT NULL,
  `d_asts` int(11) DEFAULT NULL,
  `d_pf` int(11) DEFAULT NULL,
  `d_stl` int(11) DEFAULT NULL,
  `d_to` int(11) DEFAULT NULL,
  `d_blk` int(11) DEFAULT NULL,
  `d_pts` int(11) DEFAULT NULL,
  `o_tmRebound` int(11) DEFAULT NULL,
  `d_tmRebound` int(11) DEFAULT NULL,
  `homeWon` int(11) DEFAULT NULL,
  `homeLost` int(11) DEFAULT NULL,
  `awayWon` int(11) DEFAULT NULL,
  `awayLost` int(11) DEFAULT NULL,
  `neutWon` int(11) DEFAULT NULL,
  `neutLoss` int(11) DEFAULT NULL,
  `confWon` int(11) DEFAULT NULL,
  `confLoss` int(11) DEFAULT NULL,
  `divWon` int(11) DEFAULT NULL,
  `divLoss` int(11) DEFAULT NULL,
  `pace` int(11) DEFAULT NULL,
  `won` int(11) DEFAULT NULL,
  `lost` int(11) DEFAULT NULL,
  `games` int(11) DEFAULT NULL,
  `min` int(11) DEFAULT NULL,
  `arena` varchar(255) DEFAULT NULL,
  `attendance` int(11) DEFAULT NULL,
  `bbtmID` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`year`,`tmID`),
  KEY `tmID` (`tmID`),
  KEY `tmID_2` (`tmID`,`year`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2021-02-19 14:49:36
