-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: Credit
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
-- Table structure for table `category`
--

DROP TABLE IF EXISTS `category`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `category` (
  `category_no` int(11) NOT NULL,
  `category_desc` varchar(31) NOT NULL,
  `category_code` char(2) NOT NULL,
  PRIMARY KEY (`category_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `charge`
--

DROP TABLE IF EXISTS `charge`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `charge` (
  `charge_no` int(11) NOT NULL,
  `member_no` int(11) NOT NULL,
  `provider_no` int(11) NOT NULL,
  `category_no` int(11) NOT NULL,
  `charge_dt` datetime NOT NULL,
  `charge_amt` decimal(19,4) NOT NULL,
  `statement_no` int(11) NOT NULL,
  `charge_code` char(2) NOT NULL,
  PRIMARY KEY (`charge_no`),
  KEY `charge_category_no` (`category_no`),
  KEY `charge_statement_no` (`statement_no`),
  KEY `charge_statement_no_2` (`statement_no`),
  KEY `charge_statement_no_3` (`statement_no`),
  KEY `charge_member_no` (`member_no`),
  KEY `charge_statement_no_4` (`statement_no`),
  KEY `charge_member_no_2` (`member_no`),
  KEY `charge_statement_no_5` (`statement_no`),
  KEY `charge_member_no_3` (`member_no`),
  KEY `charge_provider_no` (`provider_no`),
  CONSTRAINT `charge_ibfk_1` FOREIGN KEY (`category_no`) REFERENCES `category` (`category_no`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `charge_ibfk_2` FOREIGN KEY (`member_no`) REFERENCES `member` (`member_no`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `charge_ibfk_3` FOREIGN KEY (`member_no`) REFERENCES `member` (`member_no`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `charge_ibfk_4` FOREIGN KEY (`provider_no`) REFERENCES `provider` (`provider_no`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `corporation`
--

DROP TABLE IF EXISTS `corporation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `corporation` (
  `corp_no` int(11) NOT NULL,
  `corp_name` varchar(31) NOT NULL,
  `street` varchar(15) NOT NULL,
  `city` varchar(15) NOT NULL,
  `state_prov` char(2) NOT NULL,
  `country` char(2) NOT NULL,
  `mail_code` char(10) NOT NULL,
  `phone_no` char(13) NOT NULL,
  `expr_dt` datetime NOT NULL,
  `region_no` int(11) NOT NULL,
  `corp_code` char(2) NOT NULL,
  PRIMARY KEY (`corp_no`),
  KEY `corporation_region_no` (`region_no`),
  CONSTRAINT `corporation_ibfk_1` FOREIGN KEY (`region_no`) REFERENCES `region` (`region_no`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `member`
--

DROP TABLE IF EXISTS `member`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `member` (
  `member_no` int(11) NOT NULL,
  `lastname` varchar(15) NOT NULL,
  `firstname` varchar(15) NOT NULL,
  `middleinitial` char(1) DEFAULT NULL,
  `street` varchar(15) NOT NULL,
  `city` varchar(15) NOT NULL,
  `state_prov` char(2) NOT NULL,
  `country` char(2) NOT NULL,
  `mail_code` char(10) NOT NULL,
  `phone_no` char(13) DEFAULT NULL,
  `photograph` longblob DEFAULT NULL,
  `issue_dt` datetime NOT NULL,
  `expr_dt` datetime NOT NULL,
  `region_no` int(11) NOT NULL,
  `corp_no` int(11) DEFAULT NULL,
  `prev_balance` decimal(19,4) DEFAULT NULL,
  `curr_balance` decimal(19,4) DEFAULT NULL,
  `member_code` char(2) NOT NULL,
  PRIMARY KEY (`member_no`),
  KEY `member_region_no` (`region_no`),
  KEY `member_corp_no` (`corp_no`),
  CONSTRAINT `member_ibfk_1` FOREIGN KEY (`region_no`) REFERENCES `region` (`region_no`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `member_ibfk_2` FOREIGN KEY (`corp_no`) REFERENCES `corporation` (`corp_no`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `payment`
--

DROP TABLE IF EXISTS `payment`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `payment` (
  `payment_no` int(11) NOT NULL,
  `member_no` int(11) NOT NULL,
  `payment_dt` datetime NOT NULL,
  `payment_amt` decimal(19,4) NOT NULL,
  `statement_no` int(11) DEFAULT NULL,
  `payment_code` char(2) NOT NULL,
  PRIMARY KEY (`payment_no`),
  KEY `payment_statement_no` (`statement_no`),
  KEY `payment_member_no` (`member_no`),
  CONSTRAINT `payment_ibfk_1` FOREIGN KEY (`member_no`) REFERENCES `member` (`member_no`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `provider`
--

DROP TABLE IF EXISTS `provider`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `provider` (
  `provider_no` int(11) NOT NULL,
  `provider_name` varchar(15) NOT NULL,
  `street` varchar(15) NOT NULL,
  `city` varchar(15) NOT NULL,
  `state_prov` char(2) NOT NULL,
  `mail_code` char(10) NOT NULL,
  `country` char(2) NOT NULL,
  `phone_no` char(13) NOT NULL,
  `issue_dt` datetime NOT NULL,
  `expr_dt` datetime NOT NULL,
  `region_no` int(11) NOT NULL,
  `provider_code` char(2) NOT NULL,
  PRIMARY KEY (`provider_no`),
  KEY `provider_region_no` (`region_no`),
  CONSTRAINT `provider_ibfk_1` FOREIGN KEY (`region_no`) REFERENCES `region` (`region_no`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `region`
--

DROP TABLE IF EXISTS `region`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `region` (
  `region_no` int(11) NOT NULL,
  `region_name` varchar(15) NOT NULL,
  `street` varchar(15) NOT NULL,
  `city` varchar(15) NOT NULL,
  `state_prov` char(2) NOT NULL,
  `country` char(2) NOT NULL,
  `mail_code` char(10) NOT NULL,
  `phone_no` char(13) NOT NULL,
  `region_code` char(2) NOT NULL,
  PRIMARY KEY (`region_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `statement`
--

DROP TABLE IF EXISTS `statement`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `statement` (
  `statement_no` int(11) NOT NULL,
  `member_no` int(11) NOT NULL,
  `statement_dt` datetime NOT NULL,
  `due_dt` datetime NOT NULL,
  `statement_amt` decimal(19,4) NOT NULL,
  `statement_code` char(2) NOT NULL,
  PRIMARY KEY (`statement_no`),
  KEY `statement_member_no` (`member_no`),
  CONSTRAINT `statement_ibfk_1` FOREIGN KEY (`member_no`) REFERENCES `member` (`member_no`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `status`
--

DROP TABLE IF EXISTS `status`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `status` (
  `status_code` char(2) NOT NULL,
  `status_desc` varchar(31) NOT NULL,
  PRIMARY KEY (`status_code`)
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

-- Dump completed on 2021-02-22 17:15:59
