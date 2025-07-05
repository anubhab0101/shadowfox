justice_league = [
    "Superman",
    "Batman",
    "Wonder Woman",
    "Flash",
    "Aquaman",
    "Green Lantern"
]
print(f"Step 1 - The Justice League assembles: {justice_league}")
print(f"Step 1 - Number of heroes: {len(justice_league)}")

new_recruits = ["Batgirl", "Nightwing"]
justice_league += new_recruits
print(f"Step 2 - New recruits join the team: {justice_league}")

justice_league.remove("Wonder Woman")
justice_league.insert(0, "Wonder Woman")
print(f"Step 3 - Wonder Woman leads the League: {justice_league}")

pos_flash = justice_league.index("Flash")
pos_aquaman = justice_league.index("Aquaman")
if pos_aquaman < pos_flash:
    pos_flash, pos_aquaman = pos_aquaman, pos_flash
justice_league.remove("Green Lantern")
justice_league.insert(pos_flash + 1, "Green Lantern")
print(f"Step 4 - Green Lantern mediates between Aquaman and Flash: {justice_league}")

justice_league = [
    "Cyborg",
    "Shazam",
    "Hawkgirl",
    "Martian Manhunter",
    "Green Arrow"
]
print(f"Step 5 - A new era begins: {justice_league}")

justice_league.sort()
print(f"Step 6 - The League in order: {justice_league}")
print(f"Step 6 - The new leader is: {justice_league[0]}")
