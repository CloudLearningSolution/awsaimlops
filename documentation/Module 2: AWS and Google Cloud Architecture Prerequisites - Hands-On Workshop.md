# MLOPS Model Migration Workshop – Week 2: AWS & Google Cloud Architecture Essentials - Hands-On Workshop

---

## Module Learning Objectives

By the end of this workshop, participants will be able to:

- Compare and contrast AWS and Google Cloud global infrastructure architectures
- Design topologies across both platforms
- Map AWS services to Google Cloud service equivalents for AI/ML pipeline workloads
- Implement basic IAM configurations for secure ML environments

---

## Prerequisites

- Completion of Module 1: Cost Management Foundation
- AWS Management Console access with infrastructure permissions
- Google Cloud Console access with project access rights

---

## Workshop Overview

This hands-on workshop builds upon the cost management foundation from Module 1 to establish the technical architecture knowledge required for successful AWS to Google Cloud ML migrations. Using **console interfaces** and **CloudShell**, participants will gain practical experience with infrastructure services, networking, and IAM configurations across both platforms.

---

## Module 1: AWS Global Infrastructure and Core Resources

### Lab 2.1: AWS Regions and Availability Zones Architecture Deep Dive

- **Duration:** 45 minutes
- **Objective:** Understand AWS global infrastructure and design for high availability

### Lab Prerequisites

- AWS Management Console access with infrastructure permissions
- AWS CloudShell access enabled
- Basic understanding of cloud concepts

---

## 🧪 Hands-On Lab

---

### Part 1: Region Explorer (15 minutes)

#### ✅ Step 1: Access AWS Management Console


1. Open your web browser
2. Navigate to [https://console.aws.amazon.com](https://console.aws.amazon.com)
3. Sign in to your AWS account
4. Open CloudShell by clicking the terminal icon
5. Use the Search Bar to enter `EC2` and select the EC2 service

#### ✅ Step 2: List All Regions via CloudShell

aws ec2 describe-regions --output table

aws ec2 describe-regions --query 'Regions[*].[RegionName,OptInStatus]' --output table

aws ec2 describe-regions --query 'length(Regions)' --output text

#### Step 3: Explore Region Details via Console

1. Use the AWS Console Search Bar to enter `EC2`
2. Select the EC2 service
3. In the top-right corner, click the Region selector
4. Review available regions and note those labeled “Opt-in required”

#### Step 4: Analyze Availability Zones

1. Current region AZs
aws ec2 describe-availability-zones --output table

2. Specific region example
aws ec2 describe-availability-zones --region us-east-1 --output table

3. Count AZs per region
aws ec2 describe-regions --query 'Regions[*].RegionName' --output text | \
while read region; do
  count=$(aws ec2 describe-availability-zones --region $region \
  --query 'length(AvailabilityZones)' --output text 2>/dev/null || echo "0")
  echo "$region: $count AZs"
done

---

# Lab 2.2: AWS Edge Locations and CloudFront Global Network Exploration

---

## Duration
**30 minutes**

## Objective
Implement global content delivery and edge computing strategies.

---

## Prerequisites

- AWS Management Console access with infrastructure permissions
- AWS CloudShell access enabled
- Basic understanding of CDN concepts

---

## Theory Review

- AWS operates **450+ Points of Presence (PoPs)** and **13 Regional Edge Caches (RECs)** worldwide
- **Points of Presence (PoPs):** Edge locations for content caching closest to users
- **Regional Edge Caches (RECs):** Mid-tier caches between PoPs and origin servers
- **CloudFront** for content delivery network services
- **AWS Global Accelerator** for application performance optimization
- **Lambda@Edge** and **CloudFront Functions** for serverless compute at edge locations

---

## 🧪 Hands-On Lab

### Part 1: CloudFront Distribution Setup (15 minutes)

#### ✅ Step 1: Create S3 Bucket for Origin

**Navigate to S3 service in AWS Console**

1. Go to **S3 > Buckets**
2. Click **Create bucket**

**Configure S3 bucket**

- Bucket name: `ml-content-origin-[random-number]`
- Region: Keep default
- Block all public access: **Uncheck** (for demo purposes)
- Click **Create bucket**

**Upload sample content via Console**

1. Click your bucket name
2. Click **Upload**
3. Upload a sample image or HTML file
4. Click **Upload**

---

#### 🌐 Step 2: Create CloudFront Distribution via Console

**Navigate to CloudFront service**

1. Go to **CloudFront > Distributions**
2. Click **Create Distribution**

**Configure Origin settings**

- Origin domain: Select your S3 bucket
- Name: Keep default
- Origin access: Origin access control settings (recommended)
- Create new OAC if prompted

**Configure Default cache behavior**

- Viewer protocol policy: Redirect HTTP to HTTPS
- Allowed HTTP methods: GET, HEAD
- Cache policy: Caching Optimized

**Configure Distribution settings**

- Price class: Use all edge locations
- Default root object: `index.html` (if uploaded)
- Click **Create Distribution**

---

#### 🔍 Step 3: Verify Distribution via CloudShell

**List CloudFront distributions**


aws cloudfront list-distributions --query 'DistributionList.Items[*].[Id,DomainName,Status]' --output table

---

# Lab 2.4: Google Cloud Edge Network and Cloud CDN Exploration

---

## Duration

**30 minutes**

## Objective

Implement global content delivery using Google's edge network.

---

## Prerequisites

- Google Cloud Console access with project access rights
- Cloud Shell access enabled
- Basic understanding of CDN concepts

---

## Theory Review

- Google Cloud operates **202+ network edge locations** across 200+ countries and territories
- **Cloud CDN:** 100+ cache locations for content delivery
- **Media CDN:** 3,000+ locations for video streaming and large file downloads
- Global load balancing with **Anycast IP addresses**
- Integration with Google’s private global network backbone

---

## 🧪 Hands-On Lab

### Part 1: Cloud Storage Setup for CDN Origin (10 minutes)

#### ✅ Step 1: Create Cloud Storage Bucket via Console

**Navigate to Cloud Storage:**

1. Go to **Cloud Storage > Buckets**
2. Click **Create bucket**

**Configure Storage Bucket:**

- Name: `ml-cdn-content-[random-number]`
- Location type: Multi-region
- Default storage class: Standard
- Access control: Fine-grained
- Click **Create**

**Upload Sample Content:**

1. Click your bucket name
2. Click **Upload files**
3. Upload an image, video, or HTML file
4. Click **Upload**

**Make Content Publicly Accessible:**

1. Select the uploaded file
2. Click **Permissions** tab
3. Click **Grant access**
4. New principals: `allUsers`
5. Role: `Storage Object Viewer`
6. Click **Save**

---

### Part 2: Cloud CDN Configuration (15 minutes)

#### 🌐 Step 2: Create HTTP Load Balancer with CDN via Console

**Navigate to Load Balancing:**

1. Go to **Network Services > Load balancing**
2. Click **Create load balancer**

**Choose Load Balancer Type:**

- Select **Global external Application Load Balancer**
- Click **Configure**

**Configure Backend (Enable Cloud CDN):**

- Name: `ml-cdn-backend`
- Backend type: Cloud Storage bucket
- Cloud Storage bucket: Select your bucket
- Enable **Cloud CDN**
- Cache mode: `CACHE_ALL_STATIC`
- Default TTL: `3600 seconds`

**Configure Frontend:**

- Name: `ml-cdn-frontend`
- Protocol: HTTP
- Port: 80
- IP Address: Create IP address (`ml-cdn-ip`)

**Review and Create:**

- Name: `ml-cdn-lb`
- Click **Create**

---

#### 🔍 Step 3: Verify CDN Configuration via Cloud Shell

**List Load Balancers:**

gcloud compute url-maps list --format="table(name,defaultService)"

gcloud compute url-maps describe ml-cdn-lb --global

gcloud compute backend-services list --global --format="table(name,enableCDN,cdnPolicy.cacheMode)"

Part 3: CDN Performance Testing (5 minutes)

📊 Step 4: Test CDN Performance via Cloud Shell

Get Load Balancer IP:


LB_IP=$(gcloud compute addresses describe ml-cdn-ip --global --format="value(address)")
echo "Load Balancer IP: $LB_IP"

Test CDN Cache Behavior:

# First request (cache miss)
echo "First request (cache miss):"
curl -I http://$LB_IP/your-file.jpg | grep -E "(HTTP|Cache-Control|Age|X-Cache)"

# Second request (cache hit)
echo "Second request (cache hit):"
curl -I http://$LB_IP/your-file.jpg | grep -E "(HTTP|Cache-Control|Age|X-Cache)"

Test Global Distribution:

curl -o /dev/null -s -w "Total time: %{time_total}s\n" http://$LB_IP/your-file.jpg

🌍 Step 5: Configure Custom Domain and SSL via Console
1. **Reserve Global IP Address:**

   - Go to **VPC Network > IP addresses**
   - Click **Reserve external static address**
   - Name: `ml-cdn-ssl-ip`
   - Type: Global
   - Click **Reserve**
2. **Create SSL Certificate (Recommended: Google-managed):**
   - Go to **Network Security > SSL certificates**
   - Click **Create SSL certificate**
   - Name: `ml-cdn-ssl-cert`
   - Mode: Google-managed
   - Domains: Enter your domain name
   - Click **Create**
3. **Update Load Balancer for HTTPS:**
   - Go to **Network Services > Load balancing**
   - Click on `ml-cdn-lb`
   - Click **Edit**
   - Add new frontend:
     - Name: `ml-cdn-https-frontend`
     - Protocol: HTTPS
     - Port: 443
     - IP address: `ml-cdn-ssl-ip`
     - Certificate: `ml-cdn-ssl-cert`
   - Click **Update**
---
