import 'package:flutter/material.dart';

class RobotCharacter extends StatelessWidget {
  const RobotCharacter({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 140,
      height: 140,
      child: Stack(
        children: [
          // Main body (black, rounded)
          Positioned(
            top: 20,
            left: 20,
            right: 20,
            bottom: 40,
            child: Container(
              decoration: BoxDecoration(
                color: Colors.black,
                borderRadius: BorderRadius.circular(30),
              ),
            ),
          ),
          
          // Head with visor
          Positioned(
            top: 0,
            left: 30,
            right: 30,
            child: Container(
              height: 50,
              decoration: BoxDecoration(
                color: Colors.black,
                borderRadius: BorderRadius.circular(25),
              ),
              child: Stack(
                children: [
                  // Blue visor
                  Positioned(
                    top: 8,
                    left: 8,
                    right: 8,
                    child: Container(
                      height: 20,
                      decoration: BoxDecoration(
                        color: const Color(0xFF42A5F5), // Light blue
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          // Left eye
                          Container(
                            width: 6,
                            height: 6,
                            decoration: const BoxDecoration(
                              color: Colors.white,
                              shape: BoxShape.circle,
                            ),
                          ),
                          // Right eye
                          Container(
                            width: 6,
                            height: 6,
                            decoration: const BoxDecoration(
                              color: Colors.white,
                              shape: BoxShape.circle,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  
                  // Purple fins on sides
                  Positioned(
                    top: 15,
                    left: -5,
                    child: Container(
                      width: 8,
                      height: 20,
                      decoration: BoxDecoration(
                        color: const Color(0xFF9C27B0), // Purple
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                  Positioned(
                    top: 15,
                    right: -5,
                    child: Container(
                      width: 8,
                      height: 20,
                      decoration: BoxDecoration(
                        color: const Color(0xFF9C27B0), // Purple
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          // Purple shoulders
          Positioned(
            top: 60,
            left: 15,
            child: Container(
              width: 25,
              height: 25,
              decoration: const BoxDecoration(
                color: Color(0xFF9C27B0), // Purple
                shape: BoxShape.circle,
              ),
            ),
          ),
          Positioned(
            top: 60,
            right: 15,
            child: Container(
              width: 25,
              height: 25,
              decoration: const BoxDecoration(
                color: Color(0xFF9C27B0), // Purple
                shape: BoxShape.circle,
              ),
            ),
          ),
          
          // White arms
          Positioned(
            top: 70,
            left: 10,
            child: Container(
              width: 15,
              height: 40,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          ),
          Positioned(
            top: 70,
            right: 10,
            child: Container(
              width: 15,
              height: 40,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          ),
          
          // Chest components
          Positioned(
            top: 80,
            left: 35,
            right: 35,
            child: Column(
              children: [
                // Light blue slot
                Container(
                  height: 12,
                  decoration: BoxDecoration(
                    color: const Color(0xFF42A5F5), // Light blue
                    borderRadius: BorderRadius.circular(6),
                  ),
                ),
                const SizedBox(height: 4),
                // Purple square with AI logo
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: const Color(0xFF9C27B0), // Purple
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Center(
                    child: Text(
                      'AI',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 4),
                // Grey fins below
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    Container(
                      width: 8,
                      height: 6,
                      decoration: BoxDecoration(
                        color: Colors.grey[400],
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                    Container(
                      width: 8,
                      height: 6,
                      decoration: BoxDecoration(
                        color: Colors.grey[400],
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                    Container(
                      width: 8,
                      height: 6,
                      decoration: BoxDecoration(
                        color: Colors.grey[400],
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
} 